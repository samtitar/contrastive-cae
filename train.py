import os
import copy
import random
import engine
import argparse
import numpy as np
import data_utils.datasets as custom_datasets

import torch
import torch.nn as nn

import torchvision

from models.cae.ComplexAutoEncoder import ComplexAutoEncoder
from data_utils.transforms import DataAugmentationMomCAE, DataAugmentationPatCAE


def parse_args():
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("Set MoCAE model", add_help=False)

    # Model settings
    parser.add_argument("--image-channels", default=1, type=int)
    parser.add_argument("--image-height", default=224, type=int)
    parser.add_argument("--image-width", default=224, type=int)
    parser.add_argument("--num-features", default=1024, type=int)
    parser.add_argument("--num-segments", default=20, type=int)
    parser.add_argument("--mag-enhance", default=0.1, type=float)

    # Training settings
    parser.add_argument("--training-type", default="pre", choices=["pre", "mom", "pat"])
    parser.add_argument("--dataset", default="StanfordCars", type=str)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--teacher-momentum", default=0.996, type=float)
    parser.add_argument("--num-augments", default=8, type=int)
    parser.add_argument("--train-batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--dataset-size", default=1.0, type=float)
    parser.add_argument("--train-segnet", default="false", type=str2bool)

    # Data/machine settings
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
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
        args.mag_enhance,
    )

    if args.checkpoint_path != None:
        sd = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(sd)
    model.to(device)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    eval_transforms_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width)),
    ]

    targ_transforms_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width)),
    ]

    if args.training_type == "pre":
        train_transforms_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((args.image_height, args.image_width)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
    elif args.training_type == "mom":
        train_transforms_list = [
            DataAugmentationMomCAE(
                args.num_augments, args.image_height, args.image_width
            ),
        ]
    elif args.training_type == "pat":
        train_transforms_list = [
            DataAugmentationPatCAE(
                args.num_augments, args.image_height, args.image_width
            ),
        ]
    elif args.training_type == "lat":
        train_transforms_list = [
            DataAugmentationLatCAE(
                args.num_augments, args.image_height, args.image_width
            ),
        ]

    if args.image_channels == 1:
        train_transforms_list.append(torchvision.transforms.Grayscale())
        eval_transforms_list.append(torchvision.transforms.Grayscale())

    train_transforms = torchvision.transforms.Compose(train_transforms_list)
    eval_transforms = torchvision.transforms.Compose(eval_transforms_list)
    targ_transforms = torchvision.transforms.Compose(targ_transforms_list)

    tsk = {"root": args.dataset_root}
    esk = {"root": args.dataset_root}
    ds_source = torchvision.datasets

    if args.dataset == "StanfordCars":
        tsk["split"] = "train"
        esk["split"] = "test"
    elif args.dataset == "Caltech256":
        tsk["download"] = True
        esk["download"] = True
    elif args.dataset == "ImageFolder":
        tsk["root"] = f"{args.dataset_root}/train_images"
        esk["root"] = f"{args.dataset_root}/val_images"
    elif args.dataset == "CocoDetection":
        tsk["annFile"] = f"{args.dataset_root}/annotations/instances_train2017.json"
        esk["annFile"] = f"{args.dataset_root}/annotations/instances_val2017.json"

        tsk["root"] = f"{args.dataset_root}/train2017"
        esk["root"] = f"{args.dataset_root}/val2017"

        tsk["target_transform"] = targ_transforms
        esk["target_transform"] = targ_transforms
    elif args.dataset == "CelebAMaskHQ":
        ds_source = custom_datasets
        tsk["partition"] = "train"
        esk["partition"] = "val"

        tsk["target_transform"] = targ_transforms
        esk["target_transform"] = targ_transforms
    elif args.dataset == "CLEVRMask":
        ds_source = custom_datasets
        tsk["partition"] = "train"
        esk["partition"] = "test"

        tsk["target_transform"] = targ_transforms
        esk["target_transform"] = targ_transforms
    elif args.dataset == "Tetrominoes":
        ds_source = custom_datasets
        tsk["target_transform"] = targ_transforms
        esk["target_transform"] = targ_transforms

    train_set = getattr(ds_source, args.dataset)(transform=train_transforms, **tsk)
    eval_set = getattr(ds_source, args.dataset)(transform=eval_transforms, **esk)

    train_set_len = int(len(train_set) * args.dataset_size)
    train_set = custom_datasets.ResizedDataset(train_set, train_set_len)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_batch_size = args.train_batch_size

    if not os.path.isdir(args.outdir_path):
        os.mkdir(args.outdir_path)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    tkwargs = {}
    if args.training_type == "mom":
        teacher = ComplexAutoEncoder(
            args.image_channels,
            args.image_height,
            args.image_width,
            args.num_features,
        )

        teacher.load_state_dict(model.state_dict())
        teacher.to(device)

        for p in teacher.parameters():
            p.requires_grad = False

        tkwargs["teacher"] = teacher
        tkwargs["teacher_momentum"] = args.teacher_momentum
        tkwargs["num_augments"] = args.num_augments
    elif args.training_type == "pat":
        tkwargs["num_augments"] = args.num_augments

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
            **tkwargs,
        )

        torch.save(model.state_dict(), f"{args.outdir_path}/checkpoint{epoch}.pt")
