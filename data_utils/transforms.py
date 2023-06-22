import torch
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import ImageFilter, ImageOps


def resize_boxes(img_size, boxes, size):
    def get_size_with_aspect_ratio(img_size, size):
        w, h = img_size
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(img_size, size):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(img_size, size)

    size = get_size(img_size, size)
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, img_size))

    ratio_width, ratio_height = ratios

    scaled_boxes = boxes * torch.as_tensor(
        [ratio_width, ratio_height, ratio_width, ratio_height]
    )

    return scaled_boxes


class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        return torch.tensor([i, j, h, w]), img


class RandomCrop(transforms.RandomCrop):
    def forward(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), torch.tensor([i, j, h, w])


class RandomGaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() <= self.prob:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        return img


class RandomSolarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class DataAugmentationMomCAE:
    def __init__(self, augment_number, image_height, image_width):
        self.teacher1 = transforms.Compose(
            [
                transforms.ColorJitter(hue=0.3),
                RandomGaussianBlur(1.0),
                transforms.ToTensor(),
            ]
        )

        self.teacher2 = transforms.Compose(
            [
                transforms.ColorJitter(hue=0.1),
                RandomGaussianBlur(0.1),
                transforms.ToTensor(),
            ]
        )

        self.augment_number = augment_number
        self.student = transforms.Compose(
            [
                transforms.ColorJitter(hue=0.2),
                RandomGaussianBlur(0.5),
                transforms.ToTensor(),
            ]
        )

        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_height, image_width))

    def __call__(self, image):
        imgs, params = [], []
        t1_img = self.teacher1(image)
        t2_img = self.teacher1(image)

        imgs.append(self.resize(t1_img))
        imgs.append(self.resize(t2_img))

        for _ in range(self.augment_number):
            im = self.student(image)

            imgs.append(self.resize(im))
        return torch.stack(imgs), self.totensor(image)


class DataAugmentationPatCAE:
    def __init__(self, augment_number, image_height, image_width):
        self.augment_number = augment_number
        self.augment = transforms.Compose(
            [
                RandomGaussianBlur(0.5),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                RandomCrop((image_height // 2, image_width // 2)),
            ]
        )

        self.image_width, self.image_height = image_width, image_height

        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_height, image_width))

    def __call__(self, image):
        imgs, params = [], []
        for _ in range(self.augment_number):
            im, pa = self.augment(image)

            randi = random.randint(-self.image_height // 6, self.image_height // 6)
            randj = random.randint(-self.image_width // 6, self.image_width // 6)

            pa2 = pa + torch.tensor([randi, randj, 0, 0])
            im2 = F.crop(self.totensor(image), pa2[0], pa2[1], pa2[2], pa2[3])

            imgs.append(self.resize(im))
            params.append(pa)

            imgs.append(self.resize(im2))
            params.append(pa2)
        return torch.stack(imgs), torch.stack(params), self.totensor(image)
