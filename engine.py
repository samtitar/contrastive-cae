import random
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms

import matplotlib.pyplot as plt

from utils import *
from tqdm.auto import tqdm

from data_utils.transforms import RandomResizedCrop, resize_boxes
from torchmetrics.functional import pairwise_cosine_similarity as pcs

LOG_FREQUENCY = 25


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

        target = x

        _, reconstruction, complex_out, _ = model(x)

        loss_weights = {"bce_loss": 1}
        loss_dict = {
            "bce_loss": F.binary_cross_entropy(reconstruction, target),
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
def dum_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
    )

    model.eval()

    from sklearn.manifold import TSNE

    latents = []
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        batch_size = x.shape[0]

        l, _, _, _ = model(x.to(device).float())
        latents.append(l.abs().cpu())

    latents = torch.cat(latents).numpy()
    print(latents.mean(), latents.std(), latents.min(), latents.max())
    latents = TSNE().fit_transform(latents)

    plt.scatter(latents[:, 0], latents[:, 1])
    plt.show()


def mas_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    image_channels, image_height, image_width, num_clusters = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
        kwargs["num_clusters"],
    )

    model.masked_train()

    running_loss, n_div = 0, 0

    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        batch_size = x.shape[0]
        optimizer.zero_grad()

        x = x.to(device).float()

        _, _, complex_out, clusters = model(x)
        cluster_labels, cluster_channels = apply_kmeans(
            complex_out, n_clusters=num_clusters
        )

        channel_indices = bipartite_mask_matching(
            clusters.detach().clone().cpu(), cluster_channels.float()
        ).flatten()

        batch_indices = torch.arange(x.shape[0]).unsqueeze(1)
        batch_indices = batch_indices.repeat(1, num_clusters)
        batch_indices = batch_indices.flatten()

        loss_weights = {"bce_loss": 1}
        loss_dict = {
            "bce_loss": F.binary_cross_entropy(
                clusters.flatten(end_dim=1),
                cluster_channels[batch_indices, channel_indices].float().to(device),
            ),
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


def con_train_epoch(model, dataloader, optimizer, device, epoch, logger=None, **kwargs):
    batch_size = kwargs["batch_size"]
    image_channels, image_height, image_width = (
        kwargs["image_channels"],
        kwargs["image_width"],
        kwargs["image_height"],
    )

    model.train()

    running_loss, n_div = 0, 0
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        # Compute initial object assignments on "global" image
        _, reconstruction, complex_out, clusters = model(x)
        clusters_ind = torch.argmax(clusters, dim=1)

        for cluster in range(clusters.shape[1]):
            masks = clusters[:, cluster]

            # Reparameterization for "hard" masks
            masks_hard = torch.zeros_like(masks)
            masks_hard[clusters_ind == cluster] = 1
            masks = masks_hard - masks.detach() + masks

            masks = masks.unsqueeze(1)
            masks_inv = 1 - masks

            with torch.no_grad():
                _, _, complex_p, _ = model(x, masks=masks)
                _, _, complex_n, _ = model(x, masks=masks_inv)

            pdl = 1 - polar_distance(complex_p.angle().mean(), complex_n.angle().mean())

            loss_weights = {"bce_loss": 1, "pdl_loss": 0.1}

            loss_dict = {
                "bce_loss": F.binary_cross_entropy(reconstruction, x),
                "pdl_loss": pdl,
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
                        "train_sample": epoch * batch_size * len(dataloader)
                        + batch_idx * batch_size,
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

    for batch_idx, (x, y) in enumerate(dataloader):
        batch_size = x.shape[0]
        x = x.to(device).float()

        complex_latent, reconstruction, complex_out, clusters = model(x)
        cluster_labels, _ = apply_kmeans(complex_out, n_clusters=num_clusters)

        cluster_labels = cluster_labels.float()
        cluster_labels /= float(num_clusters)

        if logger != None:
            sam_img, pha_img, rec_img, pol_img, mas_img, cls_img = plot(
                x, reconstruction, complex_out, clusters, cluster_labels, logger.Image
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
