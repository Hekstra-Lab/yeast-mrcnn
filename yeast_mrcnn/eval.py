import os
from itertools import product
from pathlib import Path

import numpy as np
import torch
import zarr
from dask.distributed import fire_and_forget

from .util import torch_ready


def create_stores(parent_path, shape):
    path = Path(parent_path)
    os.makedirs(path)
    score_store = zarr.empty(shape, dtype="array:f4", store=path / "scores.zarr")
    data_store = zarr.empty(shape, dtype="array:f4", store=path / "mask_data.zarr")
    coord_store = zarr.empty(
        (*shape, 3), dtype="array:i4", store=path / "mask_coords.zarr"
    )
    return score_store, data_store, coord_store


def dump_to_zarr(store, inds, data):
    store[inds] = data


def dump_coords(store, inds, coords):
    for i in range(len(coords)):
        store[(*inds, i)] = coords[i]


def get_batch_inds(shape, batch_size):
    inds = np.array(list(product(*(range(x) for x in shape))))
    total = np.prod(shape)
    return inds.reshape(int(total / batch_size), batch_size, -1)


def predict(model, image_stack, output_path, client, batch_size=10, device="cuda"):
    model.eval()

    score_store, data_store, coord_store = create_stores(
        output_path, image_stack.shape[:-2]
    )

    prepared_data = torch_ready(image_stack)

    batch_inds = get_batch_inds(image_stack.shape[:-2], batch_size)

    with torch.inference_mode():
        for i, inds in enumerate(batch_inds):
            images = prepared_data.vindex[inds[:, 0], inds[:, 1]].compute()
            images = list(
                torch.as_tensor(im.copy(), dtype=torch.float32, device=device)
                for im in images
            )
            preds = model(images)
            for j, p in enumerate(preds):
                store_idx = tuple(inds[j])
                masks = p["masks"].detach().squeeze()
                mask_coords = masks.nonzero(as_tuple=True)
                mask_data = masks[mask_coords].cpu().numpy()
                f_masks_data = client.scatter(mask_data)
                fire_and_forget(
                    client.submit(dump_to_zarr, data_store, store_idx, f_masks_data)
                )

                mask_coords = np.stack(
                    tuple(m.to(torch.int32).cpu().numpy() for m in mask_coords)
                )
                f_coords = client.scatter(mask_coords)
                fire_and_forget(
                    client.submit(dump_coords, coord_store, store_idx, f_coords)
                )

                f_scores = client.scatter(p["scores"].detach().cpu().numpy())
                fire_and_forget(
                    client.submit(dump_to_zarr, score_store, store_idx, f_scores)
                )
