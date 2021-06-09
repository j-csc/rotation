BATCH_SIZE = 8
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
import collections
import itertools

import cv2
import random
from datetime import datetime
import json
import gc
import logging
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision import models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.nn.parameter import Parameter
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import resnet_encoders
from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch.encoders as smp_enc
from torchvision.models.resnet import ResNet
import torchvision as tv


from copy import deepcopy
from layers_2D import RotConv, Vector2Magnitude, VectorBatchNorm, VectorMaxPool, VectorUpsampling

import math
from torch import optim
from tqdm import tqdm
from pathlib import Path
from dataloader import AirbusShipPatchDataset, AirbusShipDataset
from streaming_dataloader import StreamingShipDataset, StreamingShipValTestDataset
import joblib
import rasterio.features
from utils import LARGE_CHIP_SIZE, CHIP_SIZE, MixedLoss, NUM_WORKERS, joint_transform, mixed_loss

class RotBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None):
        super(RotBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = RotConv(inplanes, planes, kernel_size=(3,3), padding=(1,1),stride=stride)
        self.v2m = Vector2Magnitude()
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = RotConv(planes, planes, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.v2m2 = Vector2Magnitude()
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.v2m(out)
        out = self.bn1(out)
        out_x1 = out.clone()
        out_x1 = self.relu(out_x1)

        out_x1 = self.conv2(out_x1)
        out_x1 = self.v2m2(out_x1)
        out_x1 = self.bn2(out_x1)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out_x = out_x1.clone() + identity
        out_x = self.relu(out_x)

        return out_x

class ResNetEncoder(nn.Module, EncoderMixin):

    def __init__(self,depth=5,**kwargs):
        super().__init__()
        self._depth = depth
        self._out_channels: List[int] = [3,64,64,128,256,512]
        self._in_channels = 3
        self.block = BasicBlock
        self.rotblock = RotBasicBlock
        self.inplanes = 64
        self.layers: List[int] = [3,4,6,3]
        

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
        self.rot_layer1 = self._make_rot_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_rot_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.rot_layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_rot_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(self.rotblock(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(self.rotblock(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

def main():
    #  Define model
    # smp.encoders.encoders["my_awesome_encoder"] = {
    #     "encoder": ResNetEncoder,
    #     "pretrained_settings": {
    #     },
    #     'params': {}
    # }
    # model = smp.Unet(encoder_name='my_awesome_encoder', encoder_weights=None)
    device = torch.device("cuda:%d" % 1)

    model = torch.load(f'./old_models/roteq_models/model_aug_9.pth')

    # model = torch.load('./old_models/roteq_base.pth')
    model = model.to(device)

    # Train Loader

    streaming_train_dataset = StreamingShipDataset("./data/train_df.csv", "./data", 
        large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=None,
        rotation_augmentation=False, give_mask_id=False, only_ships=True)

    train_loader = DataLoader(dataset=streaming_train_dataset, batch_size = 8, num_workers=4)

    # Val Loader

    streaming_val_dataset = StreamingShipValTestDataset("./data/val_df.csv", "./data/train_v2/", 
        large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=None,
        rotation_augmentation=False, only_ships=True)

    valid_loader = DataLoader(dataset=streaming_val_dataset, batch_size = 8, num_workers=4)

    # Model params

    criterion = MixedLoss(10.0, 2.0)
    criterion.__name__ = "MixedLoss"

    # criterion = nn.BCELoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 10):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        torch.save(model, f'./old_models/roteq_models/model_aug_{i+10}.pth')

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './old_models/best_roteq_model.pth')
            print('Model saved!')

main()