# Make a mask-rcnn model, BBBC dataloader and Adam optimizer
# and train the model for 100 epochs
# Usage
# -----
# python train_bbbc.py training_data_root output_directory device

import sys

import torch
from torch.utils.data import DataLoader, random_split

from yeast_mrcnn.datasets import BBBCDataset
from yeast_mrcnn.model import make_mrcnn
from yeast_mrcnn.train import train
from yeast_mrcnn.util import collate_fn

training_root = sys.argv[1]
output_dir = sys.argv[2]
device = sys.argv[3]

model = make_mrcnn()

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

dataset = BBBCDataset(training_root, None)

train_ds, test_ds = random_split(
    dataset, [int(0.85 * len(dataset)), int(0.15 * len(dataset))]
)

train_dataloader = DataLoader(train_ds, batch_size=4, collate_fn=collate_fn)
test_dataloader = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn)

train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    device,
    output_dir=output_dir,
    epochs=1000,
    output_every=50,
)
