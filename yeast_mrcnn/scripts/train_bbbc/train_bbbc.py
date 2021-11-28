# Make a mask-rcnn model, BBBC dataloader and Adam optimizer
# and train the model for 100 epochs
# Usage
# -----
# python train_bbbc.py training_data_root output_directory device

import sys

import torch
from torch.utils.data import DataLoader

from ._util import collate_fn
from .datasets import BBBCDataset
from .model import make_mrcnn
from .train import train

training_root = sys.argv[1]
output_dir = sys.argv[2]
device = sys.argv[3]

model = make_mrcnn()

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

dataloader = DataLoader(
    BBBCDataset(training_root, None), batch_size=4, collate_fn=collate_fn
)

train(
    model,
    dataloader,
    optimizer,
    device,
    output_dir=output_dir,
    epoch=100,
    output_every=5,
)
