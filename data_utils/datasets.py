import torch

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
