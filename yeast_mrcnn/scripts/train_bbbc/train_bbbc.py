# Make a mask-rcnn model, BBBC dataloader and Adam optimizer
# and train the model for 100 epochs
# Usage
# -----
# python train_bbbc.py training_data_root output_directory device

import sys

import torch
from torch.utils.data import DataLoader

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

dataloader = DataLoader(
    BBBCDataset(training_root, None), batch_size=4, collate_fn=collate_fn
)

train(
    model,
    dataloader,
    optimizer,
    device,
    output_dir=output_dir,
    epochs=1000,
    output_every=50,
)
