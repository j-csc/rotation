import sys,os,time

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import joblib

import scipy.ndimage
from skimage.io import imread, imsave
from skimage.transform import rotate

import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from streaming_dataloader import StreamingShipDataset, StreamingShipValTestDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import resnet_encoders
from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch.encoders as smp_enc
from torchvision.models.resnet import ResNet
import torchvision as tv

import rasterio
import fiona
import shapely.geometry
import cv2
import rasterio.features

from utils import LARGE_CHIP_SIZE, CHIP_SIZE, MixedLoss, NUM_WORKERS, joint_transform, mixed_loss
from datasets import ShipTestDataset
from tqdm import tqdm
from copy import deepcopy

def IoU(pred, targs):
    pred = (pred>0).float()
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)


parser = argparse.ArgumentParser(description='Rotbias baseline test script')
parser.add_argument('--input_fn', type=str, required=True, help='The path for testing dataset.')
parser.add_argument('--input_aug_fn', type=str, required=True, help='The path for TTA (pre-gen) dataset. Use gen_test.py.')
parser.add_argument('--model_fn', type=str, required=True, help='Path to the model file to use.')
parser.add_argument('--model_aug_fn', type=str, required=True, help='Path to the aug model file to use.')
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')

args = parser.parse_args()

""" 
Sample input: 

python3 run_test.py --input_fn ./data/test_set_rotation_aug/ --input_aug_fn ./data/test_set/ 
    --model_fn ./old_models/best_model_non_aug_it3.pth --model_aug_fn ./old_models/best_model_aug_nn.pth --gpu 0
"""
def main():
    device = torch.device("cuda:%d" % args.gpu)
    loss = MixedLoss(10.0, 2.0)
    loss.__name__ = "MixedLoss"

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')

    aug_test_ds = ShipTestDataset('./data/test_set_rotation_aug/', transform=preprocessing_fn)
    aug_test_loader = DataLoader(dataset=aug_test_ds, batch_size = 1, num_workers=4)

    test_ds = ShipTestDataset('./data/test_set/', transform=preprocessing_fn)
    test_loader = DataLoader(dataset=test_ds, batch_size = 1, num_workers=4)

    print("Testing: Non-Aug model, Non-Aug")
    model = torch.load('./old_models/best_model_non_aug_it3.pth')
    model = model.to(device)
    sum_iou = 0
    count = 0
    for i, (img, mask) in tqdm(enumerate(test_loader)):        
        if mask.sum() != 0:
            pred = (model(img.cuda()))
            pred = pred.detach().cpu().double()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] =0
            
            sum_iou += (smp.utils.metrics.IoU(threshold=0.5)(pred.squeeze(), mask.squeeze()))
            count += 1

        # if i != 0 and i % 100 == 0:
        #     print(sum_iou / count)
    print("Results: Non-Aug model, Non-Aug")
    print(sum_iou, count, (sum_iou / count))
    

    print("Testing: Non-Aug model, Aug")
    sum_iou = 0
    count = 0
    for i, (img, mask) in tqdm(enumerate(aug_test_loader)):        
        if mask.sum() != 0:
            pred = (model(img.cuda()))
            pred = pred.detach().cpu().double()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] =0
            
            sum_iou += (smp.utils.metrics.IoU(threshold=0.5)(pred.squeeze(), mask.squeeze()))
            count += 1

    print("Results: Non-Aug model, Aug")
    print(sum_iou, count, (sum_iou / count))


    print("Testing: Aug model, Non-Aug")
    model = torch.load('./old_models/best_model_aug_nn.pth')
    model = model.to(device)
    sum_iou = 0
    count = 0
    for i, (img, mask) in tqdm(enumerate(test_loader)):        
        if mask.sum() != 0:
            pred = (model(img.cuda()))
            pred = pred.detach().cpu().double()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] =0
            
            sum_iou += (smp.utils.metrics.IoU(threshold=0.5)(pred.squeeze(), mask.squeeze()))
            count += 1

    print("Results: Aug model, Non-Aug")
    print(sum_iou, count, (sum_iou / count))


    print("Testing: Aug model, Aug")
    sum_iou = 0
    count = 0
    for i, (img, mask) in tqdm(enumerate(aug_test_loader)):        
        if mask.sum() != 0:
            pred = (model(img.cuda()))
            pred = pred.detach().cpu().double()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] =0
            
            sum_iou += (smp.utils.metrics.IoU(threshold=0.5)(pred.squeeze(), mask.squeeze()))
            count += 1

    print("Results: Aug model, Aug")
    print(sum_iou, count, (sum_iou / count))

main()