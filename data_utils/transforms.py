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


class DataAugmentationMoCAE:
    def __init__(self, student_augment_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        self.teacher1 = transforms.Compose(
            [flip_and_color_jitter, RandomGaussianBlur(1.0), transforms.ToTensor()]
        )

        self.teacher2 = transforms.Compose(
            [
                flip_and_color_jitter,
                RandomGaussianBlur(0.1),
                RandomSolarization(0.2),
                transforms.ToTensor(),
            ]
        )

        self.student_augment_number = student_augment_number
        self.student = transforms.Compose(
            [
                flip_and_color_jitter,
                RandomGaussianBlur(0.5),
                RandomSolarization(0.5),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image):
        result = []
        result.append(self.teacher1(image))
        result.append(self.teacher2(image))

        for _ in range(self.student_augment_number):
            result.append(self.student(image))
        return torch.stack(result)
