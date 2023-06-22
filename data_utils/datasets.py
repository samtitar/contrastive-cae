import os
import torch
import numpy as np

from PIL import Image
from pycocotools.mask import decode
from torchvision.transforms.functional import resize


class MaskedDataset(torch.utils.data.Dataset):
    def __init__(
        self, base_dataset, mask_data, base_dataset_args=(), base_dataset_kwargs={}
    ):
        self.dataset = base_dataset(*base_dataset_args, **base_dataset_kwargs)
        self.mask_data = mask_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]

        entry_masks = []
        for mask in self.mask_data[idx]:
            mask = torch.tensor(decode(mask)).float()
            entry_masks.append(mask)
        entry_masks = torch.stack(entry_masks)

        if type(entry) == tuple:
            return entry + (entry_masks,)
        return entry, entry_masks


class Tetrominoes(torch.utils.data.Dataset):
    def __init__(self, root, transform, target_transform, partition="train"):
        self.images = np.load(f"{root}/tetrominoes_images.npy")
        self.labels = np.load(f"{root}/tetrominoes_labels.npy")

        if partition == "train":
            self.images = self.images[: int(len(self.images) * 0.9)]
            self.labels = self.labels[: int(len(self.labels) * 0.9)]
        else:
            self.images = self.images[int(len(self.images) * 0.9) :]
            self.labels = self.labels[int(len(self.labels) * 0.9) :]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        lab = self.labels[idx, :, :, :, 0].argmax(axis=0)

        img = resize(torch.tensor(img).permute(2, 0, 1), (512, 512)).permute(1, 2, 0).numpy()
        lab = resize(torch.tensor(lab).unsqueeze(0), (512, 512)).permute(1, 2, 0).numpy()
        return self.transform(img), self.transform(lab)


class CelebAMaskHQ:
    def __init__(self, root, transform, target_transform, partition):
        self.img_path = f"{root}/{partition}_img"
        self.label_path = f"{root}/{partition}_label"
        self.transform_img = transform
        self.transform_label = target_transform
        self.set = []
        self.preprocess()

        self.num_images = len(self.set)

    def preprocess(self):

        for i in range(
            len(
                [
                    name
                    for name in os.listdir(self.img_path)
                    if os.path.isfile(os.path.join(self.img_path, name))
                ]
            )
        ):
            img_path = os.path.join(self.img_path, str(i) + ".jpg")
            label_path = os.path.join(self.label_path, str(i) + ".png")
            self.set.append([img_path, label_path])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        img_path, label_path = self.set[index]
        image = Image.open(img_path)
        label = Image.open(label_path)

        return self.transform_img(image), self.transform_label(label) * 255

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class CLEVRMask:
    def __init__(self, root, transform, target_transform, partition):
        self.img_path = f"{root}/{partition}_images"
        self.label_path = f"{root}/{partition}_labels"
        self.transform_img = transform
        self.transform_label = target_transform
        self.set = []
        self.preprocess()

        self.num_images = len(self.set) - 1

    def preprocess(self):

        for i in range(
            len(
                [
                    name
                    for name in os.listdir(self.img_path)
                    if os.path.isfile(os.path.join(self.img_path, name))
                ]
            )
        ):
            img_path = os.path.join(self.img_path, str(i + 1) + ".png")
            # label_path = os.path.join(self.label_path, str(i) + ".png")
            label_path = os.path.join(self.img_path, str(i + 1) + ".png")
            self.set.append([img_path, label_path])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        img_path, label_path = self.set[index]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        return self.transform_img(image), self.transform_label(label) * 255
        # return self.transform_img(image), 1

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class ResizedDataset:
    def __init__(self, dataset, new_length):
        self.dataset = dataset
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, index):
        return self.dataset[index]
