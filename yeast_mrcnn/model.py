__all__ = [
    "make_mrcnn",
]

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def make_mrcnn():
    model = maskrcnn_resnet50_fpn(
        num_classes=2, pretrained_backbone=True, trainable_backbone_layers=5
    )
    transform = GeneralizedRCNNTransform(
        min_size=800, max_size=1333, image_mean=[0], image_std=[0]
    )
    model.transform = transform
    model.backbone.body.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    return model
