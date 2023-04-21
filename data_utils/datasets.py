import os
import torch

from PIL import Image
from pycocotools.mask import decode


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


class CelebAMaskHQ:
    def __init__(self, root, transform, partition):
        self.img_path = f"{root}/{partition}_img"
        self.label_path = f"{root}/{partition}_label"
        self.transform_img = transform
        self.transform_label = transform
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
        return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class CLEVRMask:
    def __init__(self, root, transform, partition):
        self.img_path = f"{root}/{partition}_images"
        self.label_path = f"{root}/{partition}_labels"
        self.transform_img = transform
        self.transform_label = transform
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
            img_path = os.path.join(self.img_path, str(i) + ".png")
            label_path = os.path.join(self.label_path, str(i) + ".png")
            self.set.append([img_path, label_path])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        img_path, label_path = self.set[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        return self.transform_img(image), self.transform_label(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images
