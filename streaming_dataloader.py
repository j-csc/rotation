import sys,os,time

import numpy as np
from skimage.io import imread
import pandas as pd 

import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset

from utils import rle_decode, get_mask, joint_transform

class StreamingShipDataset(IterableDataset):
    
    def __init__(self, fn, data_root_dir, num_patches=8, chip_size=256, large_chip_size=362, transform=None):
        df = pd.read_csv(fn)
        
        self.num_patches = num_patches
        self.chip_size = chip_size
        self.large_chip_size = large_chip_size
        
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

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns) # in place


        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.image_fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            img_fn = self.image_fns[idx]
            mask_fn = self.mask_fns[idx]
            mask_ids_fn = self.mask_id_fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, mask_fn, mask_ids_fn)

    def stream_chips(self):
        for (img_fn, mask_fn, mask_ids_fn) in self.stream_tile_fns():
            num_skipped_chips = 0
            
            # Read images
            img_fp = imread(img_fn)
            if self.mask_exists[idx]:
                mask_fp = imread(mask_fn)
                mask_id_fp = imread(mask_id_fn)
            else:
                mask_fp = np.zeros((768, 768), dtype=np.uint8)
                mask_id_fp = np.zeros((768, 768), dtype=np.uint8)
            
            height, width, channel = img_fp.shape
            l_height, l_width = mask_fp.shape

            assert height == l_height and width == l_width
            
            # Randomly sample NUM_PATCHES from image
            
            for i in range(self.num_patches):
                # Select the top left pixel of our chip randomly
                x = np.random.randint(0, width-self.large_chip_size)
                y = np.random.randint(0, height-self.large_chip_size)

                # Read imagery / labels
                img = None
                labels = None
                
                img = img_fp[y:y+self.chip_size, x:x+self.chip_size, :]
                mask = mask_fp[y:y+self.chip_size, x:x+self.chip_size]
                mask_id = mask_id_fp[y:y+self.chip_size, x:x+self.chip_size]
                    
                    
                print(img.shape, mask.shape, mask_id.shape)

                # Transform the imagery
                if self.image_transform is not None:
                    if self.groups is None:
                        img = self.image_transform(img)
                    else:
                        img = self.image_transform(img, group)
                else:
                    img = torch.from_numpy(img).squeeze()

                # Transform the labels
                if self.use_labels:
                    if self.label_transform is not None:
                        if self.groups is None:
                            labels = self.label_transform(labels)
                        else:
                            labels = self.label_transform(labels, group)
                    else:
                        labels = torch.from_numpy(labels).squeeze()


                # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                if self.use_labels:
                    yield img, labels
                else:
                    yield img


            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())
