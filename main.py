import sys,os,time

import numpy as np
import pandas as pd
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

  # Train Loader

  streaming_train_dataset = StreamingShipDataset("./data/train_df.csv", "./data", 
    large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=preprocessing_fn,
    rotation_augmentation=True, give_mask_id=False, only_ships=True)

  train_loader = DataLoader(dataset=streaming_train_dataset, batch_size = BATCH_SIZE, num_workers=4)

  # Val Loader

  streaming_val_dataset = StreamingShipValTestDataset("./data/val_df.csv", "./data/train_v2/", 
    large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=preprocessing_fn,
    rotation_augmentation=True, only_ships=False)

  valid_loader = DataLoader(dataset=streaming_val_dataset, batch_size = BATCH_SIZE, num_workers=4)

  # Test Loader

  # streaming_test_dataset = StreamingShipValTestDataset("./data/test_df.csv", "./data/train_v2/", 
  #   large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, transform=joint_transform, preprocessing_fn=preprocessing_fn,
  #   rotation_augmentation=False, only_ships=False)

  # test_loader = DataLoader(dataset=streaming_test_dataset, batch_size = BATCH_SIZE, num_workers=4)

  # Model params

  loss = MixedLoss(10.0, 2.0)
  loss.__name__ = "MixedLoss"

  metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
  ]

  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)

  # create epoch runners 
  # it is a simple loop of iterating over dataloader`s samples
  train_epoch = smp.utils.train.TrainEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      optimizer=optimizer,
      device=device,
      verbose=True,
  )

  valid_epoch = smp.utils.train.ValidEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      device=device,
      verbose=True,
  )

  # train model for 40 epochs

  max_score = 0

  for i in range(0, 10):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    torch.save(model, f'./aug_models_new/model_aug_{i}.pth')

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
      max_score = valid_logs['iou_score']
      torch.save(model, './best_model_aug_new.pth')
      print('Model saved!')

    # if i == 3:
    #   optimizer.param_groups[0]['lr'] = 1e-3
    #   print('Decrease decoder learning rate to 1e-3!')
        
    if i == 5:
      optimizer.param_groups[0]['lr'] = 1e-5
      print('Decrease decoder learning rate to 1e-5!')

  pass


main()