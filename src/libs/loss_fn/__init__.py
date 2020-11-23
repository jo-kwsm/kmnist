import os
from typing import Optional, List

import torch.nn as nn

from .class_weight import get_class_weight

__all__ = ["get_criterion"]


def get_criterion(
    use_class_weight: bool = False,
    train_ids: List[int] = None,
    device: Optional[str] = None,
) -> nn.Module:

    if use_class_weight:
        if train_ids is None:
            raise ValueError("you should specify train ids.")

        if device is None:
            raise ValueError("you should specify a device when you use class weight.")

        class_weight = get_class_weight(train_ids).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion
