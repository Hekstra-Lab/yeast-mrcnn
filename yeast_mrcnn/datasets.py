__all__ = [
    "YeaZDataset",
    "BBBCDataset",
]
import glob
import os

import numpy as np
import tifffile as tiff
import torch

from ._util import imread


class YeaZDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = sorted(glob.glob(root + "*im.tif"))
        self.masks = sorted(glob.glob(root + "*mask.tif"))
        assert len(self.imgs) == len(self.masks)

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = tiff.imread(img_path).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = img[None, ...]

        mask = tiff.imread(mask_path)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        img = torch.as_tensor(img)
        boxes = torch.as_tensor(boxes)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        # target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class BBBCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.subdirs = sorted(os.listdir(root))

    def __getitem__(self, idx):
        # load images and masks
        sub_path = os.path.join(self.root, self.subdirs[idx])
        img_file = os.listdir(os.path.join(sub_path, "images"))[0]
        mask_files = sorted(os.listdir(os.path.join(sub_path, "masks")))

        img_path = os.path.join(sub_path, "images", img_file)
        img = imread(img_path).astype(np.float32)[..., 0]
        img = (img - img.min()) / (img.max() - img.min())
        img = img[None, ...]

        masks = np.array(
            [imread(os.path.join(sub_path, "masks", m)) for m in mask_files]
        )

        # get bounding box coordinates for each mask
        num_objs = len(mask_files)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        img = torch.as_tensor(img)
        boxes = torch.as_tensor(boxes)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        # target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.subdirs)
