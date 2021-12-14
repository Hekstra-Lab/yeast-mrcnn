__all__ = [
    "imread",
    "collate_fn",
    "torch_masks_to_labels",
    "relabel_predicted_masks",
    "torch_ready",
]
from pathlib import Path
from typing import Union

import numpy as np
from fast_overlap import overlap
from PIL.Image import open
from scipy.optimize import linear_sum_assignment


def imread(img: Union[Path, str]) -> np.ndarray:
    return np.asarray(open(img))


def collate_fn(batch):
    return tuple(zip(*batch))


def torch_masks_to_labels(t, mask_thresh=0.5):
    if t.shape[0] == 0:
        return np.zeros(t.shape[-2:], dtype=int)
    else:
        scales = np.arange(t.shape[0])[:, None, None] + 1
        masks = t.detach().cpu().numpy().squeeze() > mask_thresh
        return np.max(scales * masks, axis=0)


def relabel_predicted_masks(predicted, ground_truth):
    overlaps = overlap(predicted, ground_truth)
    pred_idx, gt_idx = linear_sum_assignment(overlaps, maximize=True)

    relabelled = np.zeros_like(predicted)
    for old, new in zip(pred_idx, gt_idx):
        relabelled[predicted == old] = new

    return relabelled


def mean_matched_iou(pred, truth):
    """
    Given a predicted label image, match the regions to the regions in the
    ground truth array and then compute iou for each matched pair.

    Parameters
    ----------
    pred : np.ndarray of int (M,N)
        Predicted labels
    truth : np.ndarray of int (M,N)
        True Labels

    Returns
    -------
    mean_matched_iou : float in [0,1)
    """

    relabelled_pred = relabel_predicted_masks(pred, truth)
    matches = np.array(
        list(set(np.unique(relabelled_pred)[1:]) & set(np.unique(truth)[1:]))
    ).astype(int)
    intersections = overlap(relabelled_pred, truth)[matches, matches]
    unions = np.sum(
        (relabelled_pred[..., None] == matches) | (truth[..., None] == matches),
        axis=(0, 1),
    )
    return np.mean(intersections / unions)


def torch_ready(arr):
    arr_min = arr.min((-2, -1), keepdims=True)
    arr_max = arr.max((-2, -1), keepdims=True)
    normalized = (arr - arr_min) / (arr_max - arr_min)

    expanded = normalized[..., None, :, :].repeat(3, -3).astype("float32")
    return expanded
