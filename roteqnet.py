import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from copy import deepcopy
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision import models
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch.encoders as smp_enc
from torch.utils.data import Dataset, DataLoader

from torchvision.models.resnet import ResNet
import torchvision as tv
from layers_2D import RotConv, Vector2Magnitude, VectorBatchNorm, VectorMaxPool, VectorUpsampling

from utils import LARGE_CHIP_SIZE, CHIP_SIZE, MixedLoss, NUM_WORKERS, joint_transform, mixed_loss, fit, evaluate, count_parameters
from streaming_dataloader import StreamingShipDataset, StreamingShipValTestDataset

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
    # activation = sigmoid, classes = 1, batch_size = 8
    device = torch.device("cuda:%d" % 0)
        
    import gc

    gc.collect()

    torch.cuda.empty_cache()

    net = RotEqNet()
    net = net.to(device)

    # Train Loader
    BATCH_SIZE = 1
    preprocessing_fn = None

    streaming_train_dataset = StreamingShipDataset("./data/train_df.csv", "./data", 
    large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=preprocessing_fn,
    rotation_augmentation=False, give_mask_id=False, only_ships=True)

    train_loader = DataLoader(dataset=streaming_train_dataset, batch_size = BATCH_SIZE, num_workers=4)

    # Val Loader

    streaming_val_dataset = StreamingShipValTestDataset("./data/val_df.csv", "./data/train_v2/", 
    large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=preprocessing_fn,
    rotation_augmentation=False, only_ships=True)

    valid_loader = DataLoader(dataset=streaming_val_dataset, batch_size = BATCH_SIZE, num_workers=4)

    # Model Params
    loss = MixedLoss(10.0, 2.0)
    loss.__name__ = "MixedLoss"

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    batch_size = 8
    print("Model has %d paramaters" % (count_parameters(net)))

    for epoch in range(0, 10):
        # Train
        training_task_losses = []
        net.train()
        tic = time.time()

        for i, (img, mask) in tqdm(enumerate(train_loader), file=sys.stdout):
            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = net(img)

            curr_loss = loss(outputs, mask)
            training_task_losses.append(curr_loss.item())
            curr_loss.backward()
            optimizer.step()
        avg_loss = np.mean(training_task_losses)
        print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
            "", epoch, time.time() - tic, avg_loss), end=""
        )
        print("")

        torch.save(net, f"./old_models/roteq_toy/model_roteq_{epoch}.pth")




        pass



""" 
    # create epoch runners 
    train_epoch = smp.utils.train.TrainEpoch(
        net, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        net, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 10):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        torch.save(net, f'./old_models/roteq_toy/model_roteq_{i}.pth')

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(net, './best_model_roteq.pth')
            print('Model saved!')
            
        if i == 5:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
"""

        



main()