import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision import models
import segmentation_models_pytorch as smp

from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch.encoders as smp_enc

from torchvision.models.resnet import ResNet
from copy import deepcopy

import torchvision as tv

from layers_2D import RotConv, Vector2Magnitude, VectorBatchNorm, VectorMaxPool, VectorUpsampling

# Define network
class RotEqNet(nn.Module):
    def __init__(self):
        super(RotEqNet, self).__init__()

		self.main = nn.Sequential(

            # Layer 1
            x = RotConv(3, 64, [3, 3], 1, 1, n_angles=6, mode=1)(x)
            x = VectorMaxPool(2)(x)
            x = VectorBatchNorm(64)(x)

            # Layer 2
            x = RotConv(64, 128, [3, 3], 1, 1, n_angles=6, mode=2)(x)
            x = VectorMaxPool(2)(x)
            x = VectorBatchNorm(128)(x)

            # Layer 3
            x = RotConv(128, 256, [3, 3], 1, 1, n_angles=6, mode=2)(x)
            x = VectorMaxPool(2)(x)
            x = VectorBatchNorm(256)(x)

            # Layer 4
            x = RotConv(256, 512, [3, 3], 1, 1, n_angles=6, mode=2)(x)
            x = VectorMaxPool(2)(x)
            x = VectorBatchNorm(512)(x)

            # Upsampling Layer
            x = VectorUpsampling(size=256)(x)
            x = Vector2Magnitude()(x)

            # FC1
            x = nn.Conv2d(512, 4096, 1)(x)
            x = nn.BatchNorm2d(4096)(x)
            x = nn.ReLU()(x)

            # FC2
            x = nn.Dropout2d(0.7)(x)
            x = nn.Conv2d(4096, 1, 1)(x)
    )

    def forward(self, x):
        x = self.main(x)
        return x


def main():
    testNet = RotEqNet()

if '__name__' == "__main__":
    main()