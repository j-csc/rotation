import sys,os,time

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

import scipy.ndimage
from skimage.io import imread, imsave
from skimage.transform import rotate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import LARGE_CHIP_SIZE, CHIP_SIZE, MixedLoss, NUM_WORKERS, joint_transform, mixed_loss
from tqdm import tqdm

from dataloader import AirbusShipPatchDataset, AirbusShipDataset
from streaming_dataloader import StreamingShipDataset, StreamingShipValTestDataset
import joblib

import rasterio
import fiona
import shapely.geometry
import cv2
import rasterio.features

import segmentation_models_pytorch as smp

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

class ShipTestDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.image_fns = glob.glob(self.file_path + "img/*")
        self.transform = transform

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        
        fn = self.image_fns[idx].split('/')[-1]
        
        mask_fn = os.path.join(self.file_path, "mask",fn.replace("jpg", "png"))
        
        # Read image
        img = imread(self.image_fns[idx])
        mask = imread(mask_fn)
        
        if self.transform != None:
            img = self.transform(img)
        else:
            img = img / 255.0
            
        p_img = np.rollaxis(img, 2, 0).astype(np.float32)
        p_img = torch.from_numpy(p_img).squeeze()

        p_mask = mask.astype(np.int64)
        p_mask = torch.from_numpy(p_mask).unsqueeze(0)
        
        return p_img, p_mask

def main():
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    CLASSES=1
    BATCH_SIZE=8

    device = torch.device("cuda:%d" % 0)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=ACTIVATION,
        classes=CLASSES
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    loss = MixedLoss(10.0, 2.0)
    loss.__name__ = "MixedLoss"

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    device = torch.device("cuda:%d" % 0)

    aug_model = torch.load('./best_model_aug.pth')
    aug_model = aug_model.to(device)

    non_aug_model = torch.load('./best_model_non_aug.pth')
    non_aug_model = non_aug_model.to(device)

    test_epoch_aug_Unet = smp.utils.train.ValidEpoch(
        aug_model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    test_epoch_non_aug_Unet = smp.utils.train.ValidEpoch(
        non_aug_model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    aug_test_ds = ShipTestDataset('./data/test_set_rotation_aug/', transform=preprocessing_fn)
    aug_test_loader = DataLoader(dataset=aug_test_ds, batch_size = 1, num_workers=1)

    test_ds = ShipTestDataset('./data/test_set/', transform=preprocessing_fn)
    test_loader = DataLoader(dataset=test_ds, batch_size = 1, num_workers=1)

    sum_iou = 0
    count = 0

    for i, (img, mask) in tqdm(enumerate(aug_test_loader)):        
        if mask.sum() != 0:
            pred = non_aug_model(img.cuda())

            pred = pred.detach().cpu().double()
            
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] =0
            
            sum_iou += (IoU(pred.squeeze(), mask.squeeze()))
            count += 1

        if i != 0 and i % 10000 == 0:
            print(sum_iou / count)

    # print("Vanilla Unet, Aug Test")
    print(sum_iou, count, (sum_iou / count))

    # print("Aug Unet, Aug Test")
    # aa_log = test_epoch_aug_Unet.run(aug_test_loader)

    # print("Aug Unet, Vanilla Test")
    # av_log = test_epoch_aug_Unet.run(test_loader)
    
    # print("Vanilla Unet, Aug Test")
    # test_epoch_non_aug_Unet.run(aug_test_loader)
    
    # print("Vanilla Unet, Vanilla Test")
    # test_epoch_non_aug_Unet.run(test_loader)





main()