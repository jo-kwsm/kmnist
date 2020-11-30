from typing import Dict


"""
Copyright (c) 2020 yiskw713
"""


def get_cls2id_map() -> Dict[str, int]:
    cls2id_map = {"お": 0, "き": 1, "す": 2, "つ": 3, "な": 4, "は": 5, "ま": 6, "や": 7, "れ": 8, "を": 9}

    return cls2id_map

def get_id2cls_map() -> Dict[int, str]:
    cls2id_map = get_cls2id_map()
    return {val: key for key, val in cls2id_map.items()}
