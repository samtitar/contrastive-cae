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

LOG_FREQUENCY = 100


def pre_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
    )

    model.train()
    running_loss, n_div = 0, 0

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        batch_size = x.shape[0]
        optimizer.zero_grad()

        x = x.to(device).float()
        reconstruction, complex_out, _ = model(x)

        loss_weights = {"rec_loss": 1}
        loss_dict = {"rec_loss": F.mse_loss(x, reconstruction)}

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
    image_channels, image_height, image_width, num_augments = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_augments"],
    )

    teacher, teacher_momentum = kwargs["teacher"], kwargs["teacher_momentum"]

    teacher.eval()
    student.train()
    running_loss, n_div = 0, 0

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        x, ox = x[0], x[1]
        batch_size = x.shape[0]
        optimizer.zero_grad()

        x_t = x[:, :2].to(device).float().flatten(start_dim=0, end_dim=1)
        x_s = x[:, 2:].to(device).float().flatten(start_dim=0, end_dim=1)

        with torch.no_grad():
            reconstruction_t, complex_t = teacher(x_t)
        reconstruction_s, complex_s = student(x_s)

        chw = (image_channels, image_height, image_width)

        x_t = x_t.view(batch_size, 2, *chw)
        x_s = x_s.view(batch_size, num_augments, *chw)
        reconstruction_s = reconstruction_s.view(batch_size, num_augments, *chw)

        complex_t = complex_t.view(batch_size, 2, *chw).mean(dim=2)
        complex_s = complex_s.view(batch_size, num_augments, *chw).mean(dim=2)

        dist_loss = 0

        for i in range(num_augments):
            for j in range(2):
                dist_loss += cos_distance(
                    stable_angle(complex_s[:, i]), stable_angle(complex_t[:, j])
                ).mean()

        loss_weights = {"rec_loss": 1, "con_loss": 0.0001}
        loss_dict = {
            "rec_loss": F.mse_loss(x_s, reconstruction_s),
            "con_loss": dist_loss / batch_size,
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


def pat_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width, num_augments = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_augments"],
    )

    queue_len = 20
    num_negatives = 100
    temperature = 0.1
    mask_ratio = 0.005

    model.train()
    running_loss, n_div = 0, 0

    prev_views = torch.zeros(0, image_height, image_width).to(device)

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        x, c, ox = x[0], x[1], x[2]
        batch_size = x.shape[0]
        optimizer.zero_grad()

        x = x.to(device).float().flatten(start_dim=0, end_dim=1)
        c = c.to(device)

        if batch_idx < queue_len:
            with torch.no_grad():
                reconstruction, complex_out = model(x)
            prev_views = torch.cat((prev_views, complex_out.detach().mean(dim=1)))
            continue
        else:
            reconstruction, complex_out = model(x)
            prev_views = torch.cat((prev_views, complex_out.detach().mean(dim=1)))

        chw = (image_channels, image_height, image_width)
        x = x.view(batch_size, num_augments * 2, *chw)
        reconstruction = reconstruction.view(batch_size, num_augments * 2, *chw)
        complex_out = complex_out.view(batch_size, num_augments * 2, *chw).mean(dim=2)

        dist_loss = 0

        for i in range(0, num_augments * 2, 2):
            j = i + 1

            # Extract top left crop coordinates for t and s crops
            si1, sj1 = c[:, i, 0], c[:, i, 1]
            ti1, tj1 = c[:, j, 0], c[:, j, 1]

            # Compute bottom right crop doordinates for t and s crops
            si2, sj2 = si1 + image_height // 2, sj1 + image_width // 2
            ti2, tj2 = ti1 + image_height // 2, tj1 + image_width // 2

            # Compute top left and bottom right crop
            # overlap coordinates between t and s
            oi1, oj1 = torch.max(ti1, si1), torch.max(tj1, sj1)
            oi2, oj2 = torch.min(ti2, si2), torch.min(tj2, sj2)

            # Project overlap coordinates onto s crop
            soi1, soj1 = oi1 - si1, oj1 - sj1
            soi2, soj2 = ((oi2 - si1) * 2 - 1).long(), ((oj2 - sj1) * 2 - 1).long()

            # Project overlap coordinates onto t crop
            toi1, toj1 = oi1 - ti1, oj1 - tj1
            toi2, toj2 = ((oi2 - ti1) * 2 - 1).long(), ((oj2 - tj1) * 2 - 1).long()

            logits = torch.zeros(batch_size, num_negatives + 1).to(device)

            for k in range(batch_size):
                sp_slice2 = stable_angle(
                    complex_out[
                        k,
                        i,
                        soi1[k].item() * 2 : soi2[k].item(),
                        soj1[k].item() * 2 : soj2[k].item(),
                    ]
                ).flatten()

                tp_slice = stable_angle(
                    complex_out[
                        k,
                        j,
                        toi1[k].item() * 2 : toi2[k].item(),
                        toj1[k].item() * 2 : toj2[k].item(),
                    ]
                ).flatten()

                # topk = int(len(sp_slice) * mask_ratio)

                # mask = torch.zeros_like(sp_slice)
                # dist = cos_distance(sp_slice, sp_slice.mean()) / 2 + 0.5
                # mask[torch.topk(dist, topk, largest=False).indices] = 1

                mask = (
                    torch.FloatTensor(sp_slice.shape).to(device).uniform_() < mask_ratio
                )

                mask_norm = mask.sum()

                logits[k, 0] = (
                    (cos_similarity(sp_slice, tp_slice.detach()) * mask).sum()
                    / mask_norm
                    / temperature
                )

            m = np.random.randint(0, len(prev_views), size=batch_size * num_negatives)

            pred = (
                stable_angle(complex_out[:, i])
                .unsqueeze(1)
                .repeat(1, num_negatives, 1, 1)
            )

            targ = stable_angle(prev_views[m]).view(
                batch_size, num_negatives, image_height, image_width
            )

            mask = torch.FloatTensor(pred.shape).to(device).uniform_() < mask_ratio
            mask_norm = mask.sum(dim=-1).sum(dim=-1)

            logits[:, 1:] = (
                (cos_similarity(pred, targ) * mask).sum(dim=-1).sum(dim=-1)
                / mask_norm
                / temperature
            )

            dist_loss += -torch.log(F.softmax(logits, dim=1)[:, 0]).mean()

        prev_views = prev_views[batch_size * num_augments * 2 :]

        loss_weights = {"rec_loss": 1, "con_loss": 0.00001}
        loss_dict = {
            "rec_loss": F.mse_loss(x, reconstruction),
            "con_loss": dist_loss / num_augments,
        }

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
    image_channels, image_height, image_width, = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
    )

    model.eval()

    for batch_idx, (x, y) in enumerate(dataloader):
        batch_size = x.shape[0]

        x = x.to(device).float()
        reconstruction, complex_out = model(x)

        if logger != None and batch_idx == 0:
            clusters = torch.zeros(batch_size, 1, image_height, image_width)

            if len(y.shape) != 4:
                cluster_lab = apply_kmeans(complex_out, n_clusters=5)
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
            # fig = plt.figure()
            # idx = 1

            # for k in range(batch_size):
            #     fig.add_subplot(batch_size, 3, idx)
            #     plt.imshow(x[k].cpu().permute(1, 2, 0))
            #     idx += 1

            #     fig.add_subplot(batch_size, 3, idx)
            #     plt.imshow(reconstruction[k].cpu().permute(1, 2, 0))
            #     idx += 1

            #     fig.add_subplot(batch_size, 3, idx)
            #     plt.imshow(
            #         CMAP(
            #             (complex_out[k].angle().mean(dim=0).cpu() + np.pi) / (2 * np.pi)
            #         )
            #     )
            #     idx += 1

            # #     fig.add_subplot(batch_size, 6, idx)
            # #     plt.imshow(
            # #         CMAP((complex_out[k].angle()[0].cpu() + np.pi) / (2 * np.pi))
            # #     )
            # #     idx += 1

            # #     fig.add_subplot(batch_size, 6, idx)
            # #     plt.imshow(
            # #         CMAP((complex_out[k].angle()[1].cpu() + np.pi) / (2 * np.pi))
            # #     )
            # #     idx += 1

            # #     fig.add_subplot(batch_size, 6, idx)
            # #     plt.imshow(
            # #         CMAP((complex_out[k].angle()[2].cpu() + np.pi) / (2 * np.pi))
            # #     )
            # #     idx += 1

            # plt.show()
            break
