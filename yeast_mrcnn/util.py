__all__ = [
    "imread",
    "collate_fn",
    "torch_masks_to_labels",
]
from pathlib import Path
from typing import Union

import numpy as np
from PIL.Image import open


def imread(img: Union[Path, str]) -> np.ndarray:
    return np.asarray(open(img))


def collate_fn(batch):
    return tuple(zip(*batch))


def torch_masks_to_labels(t, mask_thresh=0.5):
    scales = np.arange(t.shape[0])[:, None, None] + 1
    masks = t.detach().numpy().squeeze() > mask_thresh
    return np.max(scales * masks, axis=0)
