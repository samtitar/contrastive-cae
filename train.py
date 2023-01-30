import random
import argparse
import numpy as np

import torch
import torch.nn as nn

import torchvision

from engine import pre_train_epoch, dis_train_epoch

from data_utils.datasets import NpzDataset
from models.cae.ComplexAutoEncoder import ComplexAutoEncoder


def parse_args():
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("Set NIJNtje model", add_help=False)

    # Model settings
    parser.add_argument("--image-channels", default=1, type=int)
    parser.add_argument("--image-height", default=224, type=int)
    parser.add_argument("--image-width", default=224, type=int)

    # Training settings
    parser.add_argument("--training-type", default="pre", choices=["pre", "dis"])
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--dataset", default="StanfordCars", type=str)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch-size", default=32, type=int)

    # Data/machine settings
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=8, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = None
    if args.wandb_project != None:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        logger = wandb

    device = torch.device(args.device)

    model = ComplexAutoEncoder(args.image_channels, args.image_height, args.image_width)

    if args.checkpoint_path != None:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    model.to(device)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    transforms_list = [torchvision.transforms.ToTensor()]

    if args.image_channels == 1:
        transforms_list.append(torchvision.transforms.Grayscale())

    if args.training_type == "pre":
        transforms_list.append(
            torchvision.transforms.RandomChoice(
                [
                    torchvision.transforms.RandomResizedCrop(
                        (args.image_height, args.image_width), ratio=(0.01, 2)
                    ),
                    torchvision.transforms.Resize(
                        (args.image_height, args.image_width)
                    ),
                ]
            ),
        )

    transforms = torchvision.transforms.Compose(transforms_list)

    dataset = getattr(torchvision.datasets, args.dataset)(
        root=args.dataset_root, transform=transforms
    )

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(42)

    torch.manual_seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    for epoch in range(args.epochs):
        locals()[f"{args.training_type}_train_epoch"](
            model, dataloader, optimizer, device, epoch, logger=logger
        )

        torch.save(model.state_dict(), f"{args.training_type}-trained.pt")
