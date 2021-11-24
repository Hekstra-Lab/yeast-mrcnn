__all__ = [
    "imread",
    "collate_fn",
]
from pathlib import Path
from typing import Union

import numpy as np
from PIL.Image import open


def imread(img: Union[Path, str]) -> np.ndarray:
    return np.asarray(open(img))


def collate_fn(batch):
    return tuple(zip(*batch))
