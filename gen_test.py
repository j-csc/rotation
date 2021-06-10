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
from torch.utils.data import Dataset, DataLoader, IterableDataset

from utils import LARGE_CHIP_SIZE, CHIP_SIZE,CROP_POINT, NUM_WORKERS,MixedLoss, joint_transform, mixed_loss, get_mask
from tqdm import tqdm

from dataloader import AirbusShipPatchDataset, AirbusShipDataset
from streaming_dataloader import StreamingShipDataset, StreamingShipValTestDataset
import joblib

import rasterio
import fiona
import shapely.geometry
import cv2
import rasterio.features
from PIL import Image
import segmentation_models_pytorch as smp

class GenDataset(IterableDataset):
    
    def __init__(self, fn, data_root_dir, num_patches=8, chip_size=256, large_chip_size=362, 
                 transform=joint_transform, rotation_augmentation=False, preprocessing_fn=None, only_ships=True, verbose = False):

        tr_n = np.array(pd.read_csv(fn)['0'].reset_index(drop=True))
        
        self.data_root_dir = data_root_dir

        self.segmentation_df = pd.read_csv('./data/train_ship_segmentations_v2.csv').set_index('ImageId')
        
        exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

        for el in exclude_list:
            if(el in tr_n): tr_n.remove(el)
        
        if only_ships:
            def cut_empty(names):
                return [name for name in names 
                        if(type(self.segmentation_df.loc[name]['EncodedPixels']) != float)]

            tr_n = cut_empty(tr_n)
        
        self.image_fns = tr_n
        
        self.num_patches = num_patches
        self.chip_size = chip_size
        self.large_chip_size = large_chip_size
        self.rotation_augmentation = rotation_augmentation
        self.verbose = verbose
        self.preprocessing_fn = preprocessing_fn
                        
        self.transform = transform

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.image_fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            img_fn = self.image_fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (idx, img_fn)

    def stream_chips(self):
        for (idx, img_fn) in self.stream_tile_fns():
            num_skipped_chips = 0
            
            img_path = os.path.join(self.data_root_dir, img_fn)
            
            # Read images
            img_fp = imread(img_path)
            
            # Read masks
            if img_fn in self.segmentation_df.index:
                mask_fp = get_mask(img_fn, self.segmentation_df)
            else:
                mask_fp = np.zeros((768, 768), dtype=np.uint8)
            
            height, width, channel = img_fp.shape
            l_height, l_width = mask_fp.shape

            assert height == l_height and width == l_width
            
            # Randomly sample NUM_PATCHES from image
            for i in range(self.num_patches):
                # Select the top left pixel of our chip randomly
                x = np.random.randint(0, width-self.large_chip_size)
                y = np.random.randint(0, height-self.large_chip_size)

                # Read imagery / labels
                p_img = None
                p_mask = None
            
                p_img = img_fp[y:y+self.large_chip_size, x:x+self.large_chip_size, :]
                p_mask = mask_fp[y:y+self.large_chip_size, x:x+self.large_chip_size]
                
                angles = [0,60,120,180,240,300]
                                
                for ang in range(len(angles)):
                    rotate_amount = angles[ang]
                    
                    temp_p_img = rotate(p_img, rotate_amount)
                    temp_p_mask = rotate(p_mask, rotate_amount, order=0)
                    temp_p_mask = (temp_p_mask * 255).astype(np.uint8)

                    temp_p_img = temp_p_img[CROP_POINT:CROP_POINT+CHIP_SIZE, CROP_POINT:CROP_POINT+CHIP_SIZE]
                    temp_p_mask = temp_p_mask[CROP_POINT:CROP_POINT+CHIP_SIZE, CROP_POINT:CROP_POINT+CHIP_SIZE]

                    temp_p_img = np.rollaxis(temp_p_img, 2, 0).astype(np.float32)
                    temp_p_img = torch.from_numpy(temp_p_img).squeeze()

                    temp_p_mask = temp_p_mask.astype(np.int64)
                    temp_p_mask = torch.from_numpy(temp_p_mask).unsqueeze(0)

                    yield temp_p_img, temp_p_mask, angles[ang]


            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))
        
    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())


parser = argparse.ArgumentParser(description='Rotbias pre-generating test set script')
parser.add_argument('--input_img_ids', type=str, required=True, help='The path for list of input image ids (csv).')
parser.add_argument('--input_img_dir', type=str, required=True, help='The path for stored images')
parser.add_argument('--imsave_dir', type=str, required=True, help='Path to the save test images.')
parser.add_argument('--mask_imsave_dir', type=str, required=True, help='Path to save test masks.')
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')

args = parser.parse_args()

"""
Sample input:

python3 gen_test.py --input_img_ids ./data/test_df.csv --input_img_dir ./data/train_v2/ 
    --imsave_dir /home/jason/rotation/data/test_set/img/ --mask_imsave_dir /home/jason/rotation/data/test_set/mask/
    --gpu 0

"""

def main():
    gen_test = GenDataset(args.input_img_ids, args.input_img_dir,
        num_patches=18, large_chip_size=LARGE_CHIP_SIZE, chip_size=CHIP_SIZE, 
        transform=joint_transform, preprocessing_fn=None,
        rotation_augmentation=False, only_ships=True)

    gen_test_loader = DataLoader(dataset=gen_test, batch_size = 1, num_workers=4)

    for i, (img, mask, angle) in tqdm(enumerate(gen_test_loader)):        
        img_save = np.array(img.squeeze().permute(1,2,0), dtype=np.float32)
        mask_save = np.array(mask.squeeze())
        
        im = Image.fromarray((img_save * 255).astype(np.uint8))
        mask_im = Image.fromarray(mask_save.astype(np.uint8))
        
        if angle == 0:
            # print("saved")
            im.save(f'/home/jason/rotation/data/test_set/img/{i}.jpg')
            mask_im.save(f'/home/jason/rotation/data/test_set/mask/{i}.png')
        
        im.save(f'{args.imsave_dir}{i}.jpg')
        mask_im.save(f'{args.mask_imsave_dir}{i}.png')

    pass

main()