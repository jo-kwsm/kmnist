from typing import List


"""
Copyright (c) 2020 yiskw713
"""


def get_mean(norm_value: float = 255) -> List[float]:
    # mean of imagenet
    return [123.675 / norm_value]


def get_std(norm_value: float = 255) -> List[float]:
    # std fo imagenet
    return [58.395 / norm_value]
