import sys,os,time

import numpy as np
from skimage.io import imread
import pandas as pd 

import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset
import joblib

from utils import rle_decode, get_mask, joint_transform

ship_areas_by_fn = joblib.load("./data/ship_areas_by_fn.p")

class StreamingShipDataset(IterableDataset):
    
    def __init__(self, fn, data_root_dir, num_patches=8, chip_size=256, large_chip_size=362, 
                 transform=joint_transform, preprocessing_fn=None, rotation_augmentation=False, give_mask_id=True, test=False,
                 only_ships=True, verbose = False):

        tr_n = np.array(pd.read_csv(fn)['0'].reset_index(drop=True))

        segmentation_df = pd.read_csv('./data/train_ship_segmentations_v2.csv').set_index('ImageId')
        
        exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

        for el in exclude_list:
            if(el in tr_n): tr_n.remove(el)
        
        if only_ships:
            def cut_empty(names):
                return [name for name in names 
                        if(type(segmentation_df.loc[name]['EncodedPixels']) != float)]

            tr_n = cut_empty(tr_n)

        self.num_patches = num_patches
        self.chip_size = chip_size
        self.large_chip_size = large_chip_size
        self.rotation_augmentation = rotation_augmentation
        self.preprocessing_fn = preprocessing_fn
        self.give_mask_id = give_mask_id
        self.test = test
        self.verbose = verbose
        
        self.image_fns = [
            os.path.join(data_root_dir, "train_v2/", fn)
            for fn in tr_n
        ]

        self.mask_fns = [
            os.path.join(data_root_dir, "train_v2_masks/", fn.replace(".jpg", ".png"))
            for fn in tr_n
        ]

        self.mask_id_fns = [
            os.path.join(data_root_dir, "train_v2_mask_ids/", fn.replace(".jpg", ".png"))
            for fn in tr_n
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
            mask_id_fn = self.mask_id_fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (idx, img_fn, mask_fn, mask_id_fn)

    def stream_chips(self):
        for (idx, img_fn, mask_fn, mask_id_fn) in self.stream_tile_fns():
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
                p_img = None
                p_mask = None
                p_mask_id = None
                
                p_img = img_fp[y:y+self.large_chip_size, x:x+self.large_chip_size, :]
                p_mask = mask_fp[y:y+self.large_chip_size, x:x+self.large_chip_size]
                p_mask_id = mask_id_fp[y:y+self.large_chip_size, x:x+self.large_chip_size]

                if self.test == True:
                    # Only return these for testing
                    orig = p_img

                    if self.rotation_augmentation:
                        p_img, p_mask_id = self.transform(p_img, p_mask_id, rotation_augmentation=True)
                    else:
                        p_img, p_mask_id = self.transform(p_img, p_mask_id, rotation_augmentation=False)

                    assert p_img.shape == (3,256,256) and p_mask_id.shape == (256,256)

                    yield orig, img_fn, p_img, p_mask_id

                else:
                    if self.give_mask_id == True:
                        if self.rotation_augmentation:
                            p_img, p_mask_id = self.transform(p_img, p_mask_id, rotation_augmentation=True)
                        else:
                            p_img, p_mask_id = self.transform(p_img, p_mask_id, rotation_augmentation=False)

                        assert p_img.shape == (3,256,256) and p_mask_id.shape == (1,256,256)

                        yield p_img, p_mask_id
                        
                    else:
                        if self.rotation_augmentation:
                            p_img, p_mask = self.transform(p_img, p_mask, rotation_augmentation=True, preprocessing_fn=self.preprocessing_fn)
                        else:
                            p_img, p_mask = self.transform(p_img, p_mask, rotation_augmentation=False)

                        assert p_img.shape == (3,256,256) and p_mask.shape == (1,256,256)

                        
                        yield p_img, p_mask
                

            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())


class StreamingShipValTestDataset(IterableDataset):
    
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

                if self.rotation_augmentation:
                    p_img, p_mask = self.transform(p_img, p_mask, rotation_augmentation=True, preprocessing_fn=self.preprocessing_fn)
                else:
                    p_img, p_mask = self.transform(p_img, p_mask, rotation_augmentation=False)


                assert p_img.shape == (3,256,256) and p_mask.shape == (1,256,256)

                yield p_img, p_mask


            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())