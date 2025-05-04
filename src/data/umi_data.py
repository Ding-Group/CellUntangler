from typing import Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from .vae_dataset import VaeDataset
import numpy as np


# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
class UmiDataset(Dataset):
    def __init__(self, x, y=None, transforms=None):
        if y is not None:
            assert x.shape[0] == y.shape[0], \
                ('x.shape: %s, y.shape: %s' % (x.shape, y.shape))

        self.x = x
        self.y = y

        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        data = self.x[i, :]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return torch.squeeze(torch.tensor(data)), torch.tensor(self.y[i])
        else:
            return torch.tensor(data)


class UmiSparseDataset(Dataset):
    def __init__(self, x, y=None, transforms=None):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self.y is not None:
            return self.x[i], torch.tensor(self.y[i])
        else:
            return self.x[i]


# https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
class UMIVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, in_dim:int, *args: Any, **kwargs: Any) -> None:
        super().__init__(batch_size, in_dim=in_dim, img_dims=None)

    def _seed_worker(worker_id):
        print(torch.initial_seed())
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        np.random.default_rng(worker_seed)

    def _load_synth(self, dataset: UmiDataset, train: bool = True, seed: Optional[int] = None, is_sparse: bool = False) -> DataLoader:
        if seed:
            print("Dataset seed:", seed)
            np.random.seed(seed)
            np.random.default_rng(seed)
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None

        if is_sparse:
            return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0, pin_memory=True, shuffle=train, generator=g, collate_fn=collate_fn)
        
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          num_workers=0, pin_memory=True, shuffle=train,
                          generator=g)

    def create_loaders(self, x, y, train: bool = True, seed: Optional[int] = None, is_sparse: bool = False) -> Tuple[DataLoader, DataLoader]:
        if is_sparse:
            dataset = UmiSparseDataset(x, y)
        else:
            dataset = UmiDataset(x, y)

        train_loader = self._load_synth(dataset, train=True, seed=seed, is_sparse=is_sparse)

        return train_loader


def collate_fn(batch):
    batch_x = torch.stack([torch.squeeze(torch.tensor(item[0].todense().astype(np.double))) for item in batch])
    batch_y = torch.stack([item[1] for item in batch])
    return batch_x, batch_y
