import wandb
import random
import numpy as np

import torch
import torch.nn as nn

import torchvision

from engine import pre_train_epoch, dis_train_epoch

from data_utils import transforms
from data_utils.datasets import NpzDataset
from models.cae.ComplexAutoEncoder import ComplexAutoEncoder

IMG_SIZE = 224

if __name__ == "__main__":
    wandb.init(project="contrastive-cae")

    device = torch.device("cuda")

    model = ComplexAutoEncoder(1)
    model.to(device)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    dataset = torchvision.datasets.StanfordCars(
        root="/mnt/Data/datasets/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomChoice(
                    [
                        torchvision.transforms.RandomResizedCrop(
                            IMG_SIZE, ratio=(0.01, 2)
                        ),
                        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    ]
                ),
            ]
        ),
    )

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=8,
        persistent_workers=True,
    )

    for epoch in range(100):
        pre_train_epoch(model, dataloader, optimizer, device, epoch, wandb)

        torch.save(model.state_dict(), "pretrained.pt")
