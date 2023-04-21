import random
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms

import matplotlib.pyplot as plt

from utils import *
from tqdm.auto import tqdm
from models.cae.ComplexLayers import stable_angle, get_complex_number

LOG_FREQUENCY = 25


def pre_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width, num_clusters = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_clusters"],
    )

    model.train()
    running_loss, n_div = 0, 0

    base_phase = (
        torch.linspace(0, np.pi * 2, image_width)
        .repeat(image_channels, image_height, 1)
        .to(device)
    )

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        batch_size = x.shape[0]
        optimizer.zero_grad()

        phase = base_phase.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        x = x.to(device).float()
        reconstruction, complex_out, _ = model(x, phase=phase)

        loss_weights = {"rec_loss": 1}
        loss_dict = {"rec_loss": F.mse_loss(reconstruction, x)}

        loss = sum([v * loss_weights[k] for k, v in loss_dict.items()])
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        n_div += 1

        if logger != None and batch_idx % LOG_FREQUENCY == 0 and batch_idx != 0:
            logger.log(
                dict(
                    {
                        "train_epoch": epoch,
                        "train_sample": epoch * dataloader.batch_size * len(dataloader)
                        + batch_idx * dataloader.batch_size,
                        "total_loss": running_loss / n_div,
                    },
                    **loss_dict,
                )
            )

            running_loss, n_div = 0, 0


def mom_train_epoch(
    student, dataloader, optimizer, device, epoch, logger=None, **kwargs
):
    image_channels, image_height, image_width, num_clusters, num_augments = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_clusters"],
        kwargs["num_augments"],
    )

    teacher, teacher_momentum = kwargs["teacher"], kwargs["teacher_momentum"]

    teacher.eval()
    student.train()
    running_loss, n_div = 0, 0

    base_phase = (
        torch.linspace(0, np.pi * 2, image_width)
        .repeat(image_channels, image_height, 1)
        .to(device)
    )

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        batch_size = x.shape[0]
        optimizer.zero_grad()

        x_t = x[:, :2].to(device).float().flatten(start_dim=0, end_dim=1)
        x_s = x[:, 2:].to(device).float().flatten(start_dim=0, end_dim=1)

        p_t = base_phase.unsqueeze(0).repeat(batch_size * 2, 1, 1, 1)
        p_s = base_phase.unsqueeze(0).repeat(batch_size * num_augments, 1, 1, 1)

        reconstruction_t, complex_t, _ = teacher(x_t, phase=p_t)
        reconstruction_s, complex_s, _ = student(x_s, phase=p_s)

        chw = (image_channels, image_height, image_width)
        complex_t = complex_t.view(batch_size, 2, *chw).mean(dim=2)
        complex_s = complex_s.view(batch_size, num_augments, *chw).mean(dim=2)

        loss_weights = {"rec_loss": 1, "seg_loss": 0.001}
        loss_dict = {
            "rec_loss": F.mse_loss(reconstruction_s, x_s),
            "seg_loss": torch.stack(
                [
                    polar_distance(
                        stable_angle(complex_s[:, i]), complex_t[:, j].detach().angle()
                    ).mean()
                    for i in range(num_augments)
                    for j in range(2)
                ]
            ).sum(),
        }

        loss = sum([v * loss_weights[k] for k, v in loss_dict.items()])
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        n_div += 1

        with torch.no_grad():
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data.mul_(teacher_momentum).add_(
                    (1 - teacher_momentum) * param_s.detach().data
                )

        if logger != None and batch_idx % LOG_FREQUENCY == 0 and batch_idx != 0:
            logger.log(
                dict(
                    {
                        "train_epoch": epoch,
                        "train_sample": epoch * dataloader.batch_size * len(dataloader)
                        + batch_idx * dataloader.batch_size,
                        "total_loss": running_loss / n_div,
                    },
                    **loss_dict,
                )
            )

            running_loss, n_div = 0, 0


def seg_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width, num_clusters = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_clusters"],
    )

    segment = kwargs["segm_head"]

    model.eval()
    segment.train()
    running_loss, n_div = 0, 0

    base_phase = (
        torch.linspace(0, np.pi * 2, image_width)
        .repeat(image_channels, image_height, 1)
        .to(device)
    )

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):

        y = indx_mas_to_hard(y.long().to(device), c_dim=20)
        batch_size = x.shape[0]
        optimizer.zero_grad()

        phase = base_phase.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        x = x.to(device).float()
        with torch.no_grad():
            _, complex_out, _ = model(x, phase=phase)
        phases = stable_angle(complex_out).mean(dim=1, keepdim=True)

        bin_indices = discrete_phases(complex_out.angle(), num_clusters)
        bin_indices = soft_mas_to_hard(bin_indices).bool()

        bin_probas = (
            segment(phases)
            .unsqueeze(2)
            .unsqueeze(2)
            .repeat(1, 1, image_height, image_width, 1)
        )

        segmentation = (
            bin_probas.flatten(end_dim=3)[bin_indices.flatten()]
            .view(batch_size, image_height, image_width, 20)
            .permute(0, 3, 1, 2)
        )

        segmentation_hard = soft_mas_to_hard(segmentation)

        seg_loss = 0
        for i in range(batch_size):
            idx = bipartite_mask_matching(segmentation_hard[i].cpu(), y[i].cpu())

            seg_loss += F.binary_cross_entropy(segmentation[i], y[i][idx])

        loss_weights = {"seg_loss": 1}
        loss_dict = {"seg_loss": seg_loss / batch_size}

        loss = sum([v * loss_weights[k] for k, v in loss_dict.items()])
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        n_div += 1

        if logger != None and batch_idx % LOG_FREQUENCY == 0 and batch_idx != 0:
            logger.log(
                dict(
                    {
                        "train_epoch": epoch,
                        "train_sample": epoch * dataloader.batch_size * len(dataloader)
                        + batch_idx * dataloader.batch_size,
                        "total_loss": running_loss / n_div,
                    },
                    **loss_dict,
                )
            )

            running_loss, n_div = 0, 0


@torch.no_grad()
def eval_epoch(model, dataloader, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width, num_clusters = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_clusters"],
    )

    model.eval()

    base_phase = (
        torch.linspace(0, np.pi * 2, image_width)
        .repeat(image_channels, image_height, 1)
        .to(device)
    )

    for batch_idx, (x, y) in enumerate(dataloader):
        batch_size = x.shape[0]

        phase = base_phase.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        x = x.to(device).float()
        reconstruction, complex_out, complex_components = model(x, phase=phase)

        complex_out_adj = model.phase_collapse(
            complex_components[0], complex_components[1] - phase
        )

        if logger != None and batch_idx == 0:
            clusters = torch.zeros(batch_size, 1, image_height, image_width)

            if len(y.shape) != 4:
                cluster_lab, _ = apply_kmeans(complex_out, n_clusters=num_clusters)
                cluster_lab = torch.tensor(cluster_lab).float()
            else:
                cluster_lab = y / float(y.max())

            sam_img, pha_img, rec_img, pol_img, mas_img, cls_img = plot(
                x,
                reconstruction,
                complex_out,
                clusters,
                cluster_lab,
                logger.Image,
            )

            logger.log(
                {
                    "eval_epoch": epoch,
                    "sample_image": sam_img,
                    "phase_image": pha_img,
                    "recon_image": rec_img,
                    "polar_image": pol_img,
                    "masks_image": mas_img,
                    "cluster_image": cls_img,
                }
            )

            break
        else:
            pha = complex_out.angle()
            pha_adj = complex_out_adj.angle()

            fig = plt.figure()
            idx = 1

            for i in range(batch_size):
                fig.add_subplot(batch_size, 3, idx)
                plt.imshow(x[i].numpy().transpose(1, 2, 0))
                idx += 1

                fig.add_subplot(batch_size, 3, idx)
                plt.imshow(CMAP((pha[i].mean(dim=0) + np.pi) / (2 * np.pi)))
                idx += 1

                fig.add_subplot(batch_size, 3, idx)
                plt.imshow(CMAP((pha_adj[i].mean(dim=0) + np.pi) / (2 * np.pi)))
                idx += 1

            plt.show()
            break
