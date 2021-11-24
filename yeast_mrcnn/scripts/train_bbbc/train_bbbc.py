import sys
from .model import make_mrcnn
from .train import train
from .datasets import BBBCDataset
from ._util import collate_fn
from torch.utils.data import DataLoader


training_root = sys.argv[1]
output_dir = sys.argv[2]

model = make_mrcnn()


dataloader = DataLoader(
    BBBCDataset(training_root, None), batch_size=4, collate_fn=collate_fn
)

optimizer = torch.optim.Adam(model.parameters())

train(model, dataloader, optimizer, output_dir=output_dir)
