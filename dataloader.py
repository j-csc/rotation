import sys

import numpy as np
from skimage.io import imread

import torch 
from torchvision import transforms
from torch.utils.data.dataset import IterableDataset
import albumentations as albu

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE directions

def get_mask(img_id, df):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T


class StreamingGeospatialDataset(IterableDataset):
    
    def __init__(self, imagery_fns,masks_df, chip_size=768, num_chips_per_tile=1, image_transform=None, label_transform=None, nodata_check=None, verbose=False):

        self.fns = imagery_fns
        self.masks_df = masks_df

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

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
        # NOTE: A warning, when different workers are created they will all have the same numpy random seed, however will have different torch random seeds. If you want to use numpy random functions, seed appropriately.
        #seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        #np.random.seed(seed) # when different workers spawn, they have the same numpy random seed...

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            img_fn = self.fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn)

    def stream_chips(self):
        for img_fn in self.stream_tile_fns():
            num_skipped_chips = 0

            img_fp = imread(img_fn)
            label = get_mask(img_fn, df=self.masks_df)

            height, width = img_fp.shape

            img_data = None
            label_data = None
            
            print(img_fp.shape)
            print(label.shape)
            yield img_fp, label

            # img = torch.from_numpy(img).squeeze()




            # for i in range(self.num_chips_per_tile):
            #     # Select the top left pixel of our chip randomly
            #     x = np.random.randint(0, width-self.chip_size)
            #     y = np.random.randint(0, height-self.chip_size)

            #     # Read imagery / labels
            #     img = None
            #     labels = None

            #     if self.windowed_sampling:
            #         try:
            #             img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)
            #             print(img.shape)
            #             if self.use_labels:
            #                 labels = label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
            #         except RasterioError:
            #             print("WARNING: Error reading chip from file, skipping to the next chip")
            #             continue
            #     else:
            #         img = img_data[y:y+self.chip_size, x:x+self.chip_size, :]
            #         if self.use_labels:
            #             labels = label_data[y:y+self.chip_size, x:x+self.chip_size]

            #     # Check for no data
            #     if self.nodata_check is not None:
            #         if self.use_labels:
            #             skip_chip = self.nodata_check(img, labels)
            #         else:
            #             skip_chip = self.nodata_check(img)

            #         if skip_chip: # The current chip has been identified as invalid by the `nodata_check(...)` method
            #             num_skipped_chips += 1
            #             continue

            #     # Transform the imagery
            #     if self.image_transform is not None:
            #         if self.groups is None:
            #             img = self.image_transform(img)
            #         else:
            #             img = self.image_transform(img, group)
            #     else:
            #         img = torch.from_numpy(img).squeeze()

            #     # Transform the labels
            #     if self.use_labels:
            #         if self.label_transform is not None:
            #             if self.groups is None:
            #                 labels = self.label_transform(labels)
            #             else:
            #                 labels = self.label_transform(labels, group)
            #         else:
            #             labels = torch.from_numpy(labels).squeeze()


            #     # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
            #     if self.use_labels:
            #         yield img, labels
            #     else:
            #         yield img


            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())



class StreamingStandardDataset(IterableDataset):
    
    def __init__(self, imagery_fns,masks_df, chip_size=768, num_chips_per_tile=1, image_transform=None, label_transform=None, nodata_check=None, verbose=False):

        self.fns = imagery_fns
        self.masks_df = masks_df

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingStandardDataset")

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
        # NOTE: A warning, when different workers are created they will all have the same numpy random seed, however will have different torch random seeds. If you want to use numpy random functions, seed appropriately.
        #seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        #np.random.seed(seed) # when different workers spawn, they have the same numpy random seed...

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            img_fn = self.fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn)

    def stream_chips(self):
        for img_fn in self.stream_tile_fns():
            num_skipped_chips = 0

            img_fp = imread(img_fn)
            label = get_mask(img_fn, df=self.masks_df)

            height, width = img_fp.shape

            img_data = None
            label_data = None
            
            print(img_fp.shape)
            print(label.shape)
            yield img_fp, label

            # img = torch.from_numpy(img).squeeze()



            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingStandardDataset iterator")
        return iter(self.stream_chips())
