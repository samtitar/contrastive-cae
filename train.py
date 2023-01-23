import io
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from PIL import Image
from datasets import NpzDataset
from models.cae.ComplexAutoEncoder import ComplexAutoEncoder

import matplotlib

CMAP = matplotlib.colormaps["hsv"]

wandb.init(project="contrastive-cae")

model = ComplexAutoEncoder(1, 32, 32, 256, 512)
# model = ComplexAutoEncoder(3, 64, 64, 256, 512)
model.cuda()

optim = torch.optim.AdamW(model.parameters(), lr=0.001)

dataset = NpzDataset("datasets", "MNIST_shapes", "train")
# dataset = torchvision.datasets.ImageNet(
#     root="/mnt/Data/datasets/",
#     transform=torchvision.transforms.Compose(
#         [
#             torchvision.transforms.ToTensor(),
#             # torchvision.transforms.Grayscale(),
#             torchvision.transforms.Resize((64, 64)),
#         ]
#     ),
# )

# Improve reproducibility in dataloader.
g = torch.Generator()
g.manual_seed(42)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=4,
    persistent_workers=True,
)


def get_learning_rate(step, warmup_steps, lr, lr_schedule):
    if lr_schedule:
        return lr
    else:
        return get_linear_warmup_lr(step, warmup_steps, lr, lr_schedule)


def get_linear_warmup_lr(step, warmup_steps, lr, lr_schedule):
    if step < warmup_steps:
        return lr * step / warmup_steps
    else:
        return lr


def update_learning_rate(optimizer, step, warmup_steps, lr, lr_schedule):
    lr = get_learning_rate(step, warmup_steps, lr, lr_schedule)
    optimizer.param_groups[0]["lr"] = lr
    return optimizer, lr


def plot(input_image, reconstruction, complex_output):
    img_s, img_c, img_r, img_p = [], [], [], []
    for i in range(len(input_image)):
        # img = input_image[i].cpu().numpy()[0]
        img = input_image[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        # plt.imsave(f"/home/samtitar/Pictures/sample{i}.jpg", img, cmap="gray")

        pha = complex_output.angle()[i].cpu().detach().mean(dim=0).numpy() + np.pi
        mag = complex_output.abs()[i].cpu().detach().mean(dim=0).numpy()

        # Plot color mapping.
        # pha = np.tile(np.linspace(0, 2 * np.pi, 32), (32, 1))
        # mag = np.ones_like(pha)

        pha_norm = pha / (2 * np.pi)

        pha_img = CMAP(pha_norm)
        pha_img[mag < 0.1] = [1, 1, 1, 1]

        # plt.imsave(f"/home/samtitar/Pictures/cluster{i}.jpg", pha_img)

        # rec_img = reconstruction[i].cpu().detach().numpy()[0]
        rec_img = reconstruction[i].cpu().detach().numpy().transpose(1, 2, 0)

        rec_img = (rec_img - rec_img.min()) / (rec_img.max() - rec_img.min())
        # plt.imsave(f"/home/samtitar/Pictures/recon{i}.jpg", rec_img, cmap="gray")

        pha_flt = pha.reshape(pha.shape[0] * pha.shape[1])
        mag_flt = mag.reshape(mag.shape[0] * mag.shape[1])
        pha_col = CMAP(pha_flt / (2 * np.pi))
        keep = mag_flt > 0.1

        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.scatter(pha_flt[keep], mag_flt[keep], c=pha_col[keep])

        # plt.savefig(f"/home/samtitar/Pictures/polar{i}.jpg")
        img_buf = io.BytesIO()
        plt.savefig(img_buf, bbox_inches='tight')
        pol_img = np.array(Image.open(img_buf))

        plt.clf()
        plt.close()

        img_s.append(wandb.Image(img))
        img_c.append(wandb.Image(pha_img))
        img_r.append(wandb.Image(rec_img))
        img_p.append(wandb.Image(pol_img))

    return img_s, img_c, img_r, img_p


running_loss = 0

lr = 0.001
for epoch in range(10):
    for step, (x, y) in enumerate(dataloader):
        optim.zero_grad()

        x = x.cuda().float()
        reconstruction, complex_out = model(x)

        loss = F.mse_loss(reconstruction, x)
        loss.backward()

        optim.step()

        optim, lr = update_learning_rate(optim, step * (epoch + 1), 500, lr, True)

        running_loss += loss.item()

        if step % 100 == 0:
            img_s, img_c, img_r, img_p = plot(x, reconstruction, complex_out)

            # print(img_s.shape, img_c.shape, img_r.shape, img_p.shape)
            wandb.log(
                {
                    "MSE Loss": running_loss / 100,
                    "Sample Image": img_s,
                    "Phase Image": img_c,
                    "Reconstruction Image": img_r,
                    "Polar Projection": img_p,
                }
            )

            running_loss = 0

            print(f"LOSS: {loss.item()}")
