import sys
import os
import time
import datetime
import argparse
import copy

import numpy as np
import pandas as pd

from dataloader import StreamingGeospatialDataset

import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def main():
  PATH = './'
  TRAIN = 'train_v2/'
  TEST = 'test_v2/'
  SEGMENTATION = 'train_ship_segmentations_v2.csv'
  exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                  '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                  'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                  'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] 

  tr_arr = np.array(pd.read_csv('train_df.csv')['0'].reset_index(drop=True))
  val_arr = np.array(pd.read_csv('val_df.csv')['0'].reset_index(drop=True))
  test_arr = np.array(pd.read_csv('test_df.csv')['0'].reset_index(drop=True))

  for el in exclude_list:
    if(el in tr_arr): tr_arr.remove(el)
    if(el in val_arr): val_arr.remove(el)
    if(el in test_arr): test_arr.remove(el)

  segmentation_df = pd.read_csv(os.path.join(PATH, SEGMENTATION)).set_index('ImageId')

  tr_n = tr_arr
  val_n = val_arr
  test_n = test_arr


  # IF CUT_EMPTY
  def cut_empty(names):
    return [name for name in names 
            if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

  tr_n = cut_empty(tr_n)
  val_n = cut_empty(val_n)
  test_n = cut_empty(test_n)

  pass


main()