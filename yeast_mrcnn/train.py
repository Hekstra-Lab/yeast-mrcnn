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
    model, dataloader, optimizer, device, epochs=100, output_every=5, output_dir="./"
):
    big_df = pd.DataFrame()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for e in range(epochs):

        loss_df = train_one_epoch(model, dataloader, optimizer, e, device)
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
        big_df = big_df.append(loss_df)

    big_df.to_csv(output_dir + "loss_log.csv")
    return big_df


def evaluate_test(model, dataloader):

    pass
