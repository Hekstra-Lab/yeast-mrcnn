__all__ = [
    "train_one_epoch",
    "train",
    "evaluate_test",
]

import os
import sys

import numpy as np
import pandas as pd
import torch

from .validation import matched_box_iou, matched_mask_iou


def train_one_epoch(model, dataloader, optimizer, epoch, device):

    loss_df = pd.DataFrame()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # model(images, targets)
        scalar_loss_dict = {k: v.item() for k, v in loss_dict.items()}

        loss_df = loss_df.append([scalar_loss_dict])

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    return loss_df.reset_index(drop=True)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs=100,
    output_every=5,
    output_dir="./",
):
    big_df = pd.DataFrame()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.train()

    for e in range(epochs):

        loss_df = train_one_epoch(model, train_dataloader, optimizer, e, device)

        if (e + 1) % output_every == 0 or (e + 1) == epochs:
            torch.save(model.state_dict(), output_dir + f"model_state_epoch_{e+1}.pt")
            print(
                f"[Epoch {e+1}] "
                + " ".join(
                    f"{loss_name[5:]}={val:4g}"
                    for loss_name, val in loss_df.mean().iteritems()
                ),
                flush=True,
            )

        loss_df["epoch"] = e
        loss_df = loss_df.join(evaluate_test(model, val_dataloader, device))
        big_df = big_df.append(loss_df)

    big_df.to_csv(output_dir + "loss_log.csv")
    return big_df


def evaluate_test(model, dataloader, device):
    model.eval()  # cant use no_grad here bc of optimization in validation
    mask_ious = []
    box_ious = []
    for i, (images, targets) in enumerate(dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        pred_mask = predictions[0]["masks"]
        pred_boxes = predictions[0]["boxes"]
        true_mask = targets[0]["masks"]
        true_boxes = targets[0]["boxes"]

        mask_ious.append(matched_mask_iou(pred_mask, true_mask))
        box_ious.append(matched_box_iou(pred_boxes, true_boxes))

    model.train()

    out = pd.DataFrame(
        [[np.mean(mask_ious), np.std(mask_ious), np.mean(box_ious), np.std(box_ious)]],
        columns=["mmask-iou-mean", "mmask-iou-std", "mbox-iou-mean", "mbox-iou-std"],
    )
    return out
