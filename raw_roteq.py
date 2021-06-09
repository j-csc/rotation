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

if __name__ == '__main__':
    # Define network
    class Net(nn.Module):
        def __init__(self, num_input_channels, num_output_classes, num_filters=64):
            super(Net, self).__init__()
            self.main = nn.Sequential(
                RotConv(num_input_channels, num_filters, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
                VectorMaxPool(2),
                VectorBatchNorm(num_filters),

                RotConv(num_filters, 2 * num_filters, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
                VectorMaxPool(2),
                VectorBatchNorm(2 * num_filters),

                # RotConv(2* num_filters, 3 * num_filters, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
                # VectorMaxPool(2),
                # VectorBatchNorm(3 * num_filters),

                # RotConv(3* num_filters, 4 * num_filters, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
                Vector2Magnitude(),

                # nn.Conv2d(1024, 512, 1),  # FC1
                # nn.BatchNorm2d(512),
                # nn.ReLU(),
                # nn.Dropout2d(0.7),
                # nn.Conv2d(512, num_output_classes, kernel_size=1, stride=1, padding=0) # FC2
            )

        def forward(self, x):
            print(x.shape)
            x = self.main(x)
            print(x[0].shape)
            x = x.view(x.size()[0], x.size()[1])

            return x
    
    net = Net(num_input_channels=3, num_output_classes=1, num_filters=64)
    device = torch.device("cuda:%d" % 1)
    net = net.to(device)

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

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        net, 
        loss=criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        net, 
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
        
        torch.save(net, f'./raw_roteq_nets/net_aug_{i}.pth')

        # do something (save net, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(net, './raw_roteq_nets/best_roteq_net.pth')
            print('net saved!')