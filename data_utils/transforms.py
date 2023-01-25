import torch
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


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
        img = F.resized_crop(
            img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias
        )

        return torch.tensor([i, j, h, w]), img
