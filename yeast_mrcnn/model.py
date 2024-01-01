__all__ = [
    "make_mrcnn",
    "mrcnn",
]

import torch
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def make_mrcnn():
    model = maskrcnn_resnet50_fpn(
        num_classes=2, pretrained_backbone=True, trainable_backbone_layers=5
    )
    transform = GeneralizedRCNNTransform(
        min_size=800, max_size=1333, image_mean=[0], image_std=[1]
    )
    model.transform = transform
    model.backbone.body.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    return model


def mrcnn():
    # Get a resnet50 fpn backbone and change the first layer for grayscale
    backbone = resnet_fpn_backbone("resnet50", pretrained=True, trainable_layers=5)
    backbone.body.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    # Make anchor generator with 3 sizes per feature map and 5 aspect ratios
    sizes = tuple(2.0**x for x in range(5, 12))
    aspects = tuple(0.5 * x for x in range(1, 5))
    n_feature_maps = 5  # true for resnet50 with FPN
    ag_sizes = tuple(tuple(sizes[i : i + 3]) for i in range(n_feature_maps))
    ag_aspects = n_feature_maps * (aspects,)
    anchor_generator = AnchorGenerator(sizes=ag_sizes, aspect_ratios=ag_aspects)

    # Assemble into MaskRCNN
    mrcnn = MaskRCNN(
        backbone,
        2,
        image_mean=[0],
        image_std=[1],
        rpn_anchor_generator=anchor_generator,
    )

    return mrcnn
