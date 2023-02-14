import os
import random
import engine
import argparse
import numpy as np

import torch
import torch.nn as nn

import torchvision

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
    parser.add_argument("--num-features", default=1024, type=int)
    parser.add_argument("--num-clusters", default=20, type=int)

    # Training settings
    parser.add_argument(
        "--training-type", default="pre", choices=["pre", "con", "mas", "dum"]
    )
    parser.add_argument("--dataset", default="StanfordCars", type=str)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--train-batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)

    # Data/machine settings
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--outdir-path", default="checkpoints", type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = None
    if args.wandb_project != None:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update(vars(args))
        logger = wandb

        args.outdir_path = args.outdir_path.replace("[wandb_run_id]", wandb.run.id)

    device = torch.device(args.device)

    model = ComplexAutoEncoder(
        args.image_channels,
        args.image_height,
        args.image_width,
        args.num_features,
        args.num_clusters,
    )

    if args.checkpoint_path != None:
        sd = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(sd)
    model.to(device)

    if args.training_type == "mas":
        for n, p in model.named_parameters():
            if "clustering" not in n:
                p.requires_grad = False

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_transforms_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width)),
    ]

    eval_transforms_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width)),
    ]

    if args.image_channels == 1:
        train_transforms_list.append(torchvision.transforms.Grayscale())
        eval_transforms_list.append(torchvision.transforms.Grayscale())

    train_transforms = torchvision.transforms.Compose(train_transforms_list)
    eval_transforms = torchvision.transforms.Compose(eval_transforms_list)

    train_set_kwargs = {}
    eval_set_kwargs = {}

    if args.dataset == "Caltech256":
        train_set_kwargs["download"] = True
        eval_set_kwargs["download"] = True
    elif args.dataset == "CocoDetection":
        train_set_kwargs["annFile"] = f"{args.dataset_root}/annotations/train2017.json"
        eval_set_kwargs["annFile"] = f"{args.dataset_root}/annotations/val2017.json"

    train_set = getattr(torchvision.datasets, args.dataset)(
        root=args.dataset_root, transform=train_transforms, **train_set_kwargs
    )

    eval_set = getattr(torchvision.datasets, args.dataset)(
        root=args.dataset_root, transform=eval_transforms, **eval_set_kwargs
    )

    g = torch.Generator()
    g.manual_seed(42)

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_batch_size = args.train_batch_size

    if not os.path.isdir(args.outdir_path):
        os.mkdir(args.outdir_path)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    for epoch in range(args.epochs):
        engine.eval_epoch(
            model,
            eval_loader,
            device,
            epoch,
            logger=logger,
            batch_size=args.train_batch_size,
            image_channels=args.image_channels,
            image_height=args.image_height,
            image_width=args.image_width,
            num_clusters=args.num_clusters,
        )

        getattr(engine, f"{args.training_type}_train_epoch")(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            logger=logger,
            batch_size=args.train_batch_size,
            image_channels=args.image_channels,
            image_height=args.image_height,
            image_width=args.image_width,
            num_clusters=args.num_clusters,
        )

        torch.save(model.state_dict(), f"{args.outdir_path}/checkpoint{epoch}.pt")
