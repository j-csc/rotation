import sys,os,time

import numpy as np
from skimage.io import imread
import pandas as pd 

import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as albu

from utils import rle_decode, get_mask, joint_transform

class AirbusShipDataset(Dataset):

    def __init__(self, fn, data_root_dir, transform=None):
        df = pd.read_csv(fn)
        
        self.image_fns = [
            os.path.join(data_root_dir, "train_v2/", fn)
            for fn in df["0"].values
        ]

        self.mask_fns = [
            os.path.join(data_root_dir, "train_v2_masks/", fn.replace(".jpg", ".png"))
            for fn in df["0"].values
        ]

        self.mask_id_fns = [
            os.path.join(data_root_dir, "train_v2_mask_ids/", fn.replace(".jpg", ".png"))
            for fn in df["0"].values
        ]
        
        self.mask_exists = [
            os.path.exists(fn)
            for fn in self.mask_fns
        ]
        
        self.transform = transform

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):

        img_fn, mask_fn, mask_id_fn = self.image_fns[idx], self.mask_fns[idx], self.mask_id_fns[idx]
        fn = os.path.basename(img_fn)
        
        
        image = imread(img_fn)
        if self.mask_exists[idx]:
            mask = imread(mask_fn)
            mask_id = imread(mask_id_fn)
        else:
            mask = np.zeros((768, 768), dtype=np.uint8)
            mask_id = np.zeros((768, 768), dtype=np.uint8)
        
        #if self.transform:
        #    sample = self.transform(sample)

        return (image, mask, mask_id, fn)


class AirbusShipPatchDataset(Dataset):
    
    """
        Returns 8 256x256 patches per tile
    """

    def __init__(self, fn, data_root_dir, large_chip_size=362, chip_size=256, 
                 transform=joint_transform, rotation_augmentation=False, give_mask_id=True):
        df = pd.read_csv(fn)
        
        self.image_fns = [
            os.path.join(data_root_dir, "train_v2/", fn)
            for fn in df["0"].values
        ]

        self.mask_fns = [
            os.path.join(data_root_dir, "train_v2_masks/", fn.replace(".jpg", ".png"))
            for fn in df["0"].values
        ]

        self.mask_id_fns = [
            os.path.join(data_root_dir, "train_v2_mask_ids/", fn.replace(".jpg", ".png"))
            for fn in df["0"].values
        ]
        
        self.mask_exists = [
            os.path.exists(fn)
            for fn in self.mask_fns
        ]
        
        self.transform = transform
        self.rotation_augmentation = rotation_augmentation
        self.large_chip_size = large_chip_size
        self.chip_size = chip_size
        self.give_mask_id = give_mask_id

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):

        img_fn, mask_fn, mask_id_fn = self.image_fns[idx], self.mask_fns[idx], self.mask_id_fns[idx]
        fn = os.path.basename(img_fn)
        
        
        image = imread(img_fn)
        if self.mask_exists[idx]:
            mask = imread(mask_fn)
            mask_id = imread(mask_id_fn)
        else:
            mask = np.zeros((768, 768), dtype=np.uint8)
            mask_id = np.zeros((768, 768), dtype=np.uint8)
        
        res = []
        res_masks = []
        
        width, height, channel = image.shape
        
        # Extract patches
        for patch_i in range(8):
            # Select the top left pixel of our chip randomly
            x = np.random.randint(0, width-self.large_chip_size)
            y = np.random.randint(0, height-self.large_chip_size)

            # Read imagery / labels
            p_img = None
            p_mask = None
            p_mask_id = None

            p_img = image[y:y+self.large_chip_size, x:x+self.large_chip_size, :]
            p_mask = mask[y:y+self.large_chip_size, x:x+self.large_chip_size]
            p_mask_id = mask_id[y:y+self.large_chip_size, x:x+self.large_chip_size]
            
            if self.give_mask_id == True:
                if self.rotation_augmentation:
                    p_img, p_mask_id = self.transform(p_img, p_mask_id, rotation_augmentation=True)
                else:
                    p_img, p_mask_id = self.transform(p_img, p_mask_id, rotation_augmentation=False)

                assert p_img.shape == (3,256,256) and p_mask_id.shape == (256,256)

                res.append(p_img)
                res_masks.append(p_mask_id)
            else:
                if self.rotation_augmentation:
                    p_img, p_mask = self.transform(p_img, p_mask, rotation_augmentation=True)
                else:
                    p_img, p_mask = self.transform(p_img, p_mask, rotation_augmentation=False)

                assert p_img.shape == (3,256,256) and p_mask.shape == (256,256)

                res.append(p_img)
                res_masks.append(p_mask)

        return (image, mask, mask_id, fn, res, res_masks)