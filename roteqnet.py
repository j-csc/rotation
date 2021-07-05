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
            RotConv(3, 64, [3, 3], 1, 1, n_angles=6, mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(64),

            # Layer 2
            RotConv(64, 128, [3, 3], 1, 1, n_angles=6, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(128),

            # Layer 3
            RotConv(128, 256, [3, 3], 1, 1, n_angles=6, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(256),

            # Layer 4
            RotConv(256, 512, [3, 3], 1, 1, n_angles=6, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(512),

            # UpSampling Layer
            VectorUpsampling(size=256),
            Vector2Magnitude(),
            
            # FC1
            nn.Conv2d(512, 4096, 1),  
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            
            # FC2
            nn.Dropout2d(0.7),
            nn.Conv2d(4096, 1, 1)
        )

    def forward(self, x):
        x = self.main(x)
        return x

def main():
    net = RotEqNet()
    criterion = nn.BCELoss()
    

main()