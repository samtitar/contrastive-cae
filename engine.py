import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from utils import plot, polar_distance_loss

from data_utils.transforms import RandomResizedCrop, resize_boxes
from torchmetrics.functional import pairwise_cosine_similarity as pcs

LOG_FREQUENCY = 250
IMG_SIZE = 224
PBATCH_SIZE = 16


def pre_train_epoch(model, dataloader, optimizer, device, epoch, logger=None):
    running_loss, n_div = 0, 0

    for step, (x, y) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        x = x.to(device).float()
        _, reconstruction, complex_out = model(x)

        loss = F.binary_cross_entropy(reconstruction, x)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        n_div += 1

        if logger != None and step % LOG_FREQUENCY == 0:
            img_s, img_c, img_r, img_p = plot(
                x, reconstruction, complex_out, logger.Image
            )

            logger.log(
                {
                    "Loss": running_loss / n_div,
                    "Sample Image": img_s,
                    "Phase Image": img_c,
                    "Reconstruction Image": img_r,
                    "Polar Projection": img_p,
                }
            )

    running_loss, n_div = 0, 0


def dis_train_epoch(model, dataloader, optimizer, device, epoch, logger=None):
    running_loss = 0
    rrc = RandomResizedCrop(IMG_SIZE, scale=(0.1, 0.3), ratio=(0.75, 1.25))

    for step, (x, y) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        x_g = F_transforms.resize(x, (IMG_SIZE, IMG_SIZE)).to(device).float()

        for patch_idx in range(5):
            # Compute initial object assignments on "global" image
            _, _, complex_out_g = model(x_g)

            # Divide image into randomly sampled "local" patches
            boxes, x_l = list(zip(*[rrc(x[0]) for _ in range(PBATCH_SIZE)]))

            x_l = torch.stack(x_l).to(device).float()

            boxes = resize_boxes(x.shape[2:], torch.stack(boxes), (IMG_SIZE, IMG_SIZE))
            boxes[:, 2:] += boxes[:, :2]
            boxes = boxes.long()

            # Obtain features for each patch
            complex_latent_l, _, _ = model(x_l)
            features_l = complex_latent_l.abs().flatten(start_dim=1)

            # Compute similarities between features of each patch
            similarities = pcs(features_l)
            similarities.fill_diagonal_(0.5)
            similarities = (similarities - similarities.min()) / (
                similarities.max() - similarities.min()
            )

            # idx = 1

            # # Plot patches
            # fig = plt.figure()
            # fig.add_subplot(3, PBATCH_SIZE + 1, idx)
            # plt.imshow(x[0].permute(1, 2, 0), cmap="gray")
            # idx += 1

            # for i, patch in enumerate(x_l):
            #     fig.add_subplot(3, PBATCH_SIZE + 1, idx)
            #     plt.imshow(patch.permute(1, 2, 0).cpu().numpy(), cmap="gray")
            #     plt.title(f"Image: {i}")
            #     idx += 1

            # print(similarities.min(), similarities.max())

            # # Plot boxes
            # fig.add_subplot(3, PBATCH_SIZE + 1, idx)
            # plt.imshow(x_g[0].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            # idx += 1

            # for i, box in enumerate(boxes):
            #     fig.add_subplot(3, PBATCH_SIZE + 1, idx)
            #     box = x_g[0, :, box[0] : box[2], box[1] : box[3]]
            #     plt.imshow(box.permute(1, 2, 0).cpu().numpy(), cmap="gray")
            #     plt.title(f"Image: {i}")
            #     idx += 1

            # fig.add_subplot(3, PBATCH_SIZE + 1, idx)
            # plt.imshow(similarities.cpu().detach().numpy())
            # plt.show()

            loss = polar_distance_loss(
                complex_out_g[0, 0].angle(), boxes, similarities.to(device)
            )

            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        with torch.no_grad():
            complex_latent, reconstruction, complex_out = model(x_g)

        if logger != None and step % LOG_FREQUENCY == 0:
            img_s, img_c, img_r, img_p = plot(
                x, reconstruction, complex_out, logger.Image
            )

            logger.log(
                {
                    "MSE Loss": running_loss / 100,
                    "Sample Image": img_s,
                    "Phase Image": img_c,
                    "Reconstruction Image": img_r,
                    "Polar Projection": img_p,
                }
            )

            running_loss = 0
