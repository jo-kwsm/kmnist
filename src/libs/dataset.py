import os
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from libs.class_id_map import get_id2cls_map

__all__ = ["get_dataloader"]

def get_dataloader(
    imgs: List[int],
    ids: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:

    data = KMnistDataset(
        imgs,
        ids,
        transform=transform,
    )

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class KMnistDataset(Dataset):
    def __init__(
        self, imgs: List[int], ids: List[int], transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()

        self.imgs = imgs
        self.ids = ids
        self.n_classes = 10
        self.transform = transform
        self.id2cls_map = get_id2cls_map()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self.imgs[idx]
        img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)

        cls_id = self.ids[idx]
        label = self.id2cls_map[cls_id]
        cls_id = torch.tensor(cls_id).long()

        sample = {"img": img, "class_id": cls_id, "label": label, "img_path": idx}

        return sample

    def get_n_classes(self) -> int:
        return self.n_classes
