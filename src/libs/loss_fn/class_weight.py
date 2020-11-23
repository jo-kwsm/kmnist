import pandas as pd
import numpy as np
import torch

from typing import List


def get_class_num(train_ids: List[int]) -> torch.Tensor:
    """
    get the number of samples in each class
    Args:
        train_ids: list of train ids
    """

    n_classes = len(np.unique(train_ids))

    nums = {}
    for i in range(n_classes):
        nums[i] = 0
    for v in train_ids:
        nums[v] += 1
    class_num = []
    for val in nums.values():
        class_num.append(val)
    class_num = torch.tensor(class_num)

    return class_num


def get_class_weight(train_ids: List[int]) -> torch.Tensor:
    """Class weight for CrossEntropy in EASI score Dataset Class
    weight is calculated in the way described in:
    D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels
    with a common multi-scale convolutional architecture,” in ICCV 2015,
    openaccess:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    class_num = get_class_num(train_ids)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight
