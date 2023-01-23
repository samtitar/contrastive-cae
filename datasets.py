import torch

import numpy as np

from pathlib import Path

class NpzDataset(torch.utils.data.Dataset):
    """NpzDataset: loads a npz file as input."""

    def __init__(self, root, filename, partition):
        filepath = Path(root, f"{filename}_{partition}.npz")

        self.dataset = np.load(filepath)
        self.images = torch.Tensor(self.dataset["images"])
        self.labels = self.dataset["labels"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images = (self.images[idx] + 1) / 2  # Normalize to [0, 1] range.
        return images, self.labels[idx]