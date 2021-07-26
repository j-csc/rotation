# Rotation Bias Baseline

This repository contains implementations of several baseline experiments and data visualizations for "An exploration of rotation bias in geospatial datasets".

(Paper soon)

## Requirements

To install requirements:

```setup
git clone https://github.com/jaloo555/rotation.git
cd rotation
conda create -n rotbias
conda activate rotbias
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install matplotlib pandas numpy rasterio fiona segmentation_models_pytorch shapely scikit-image
pip install kaggle
```

To download datasets:

```download
# For Airbus Ship detection dataset
kaggle competitions download -c airbus-ship-detection
```


## Project description 

|           File          |                                   Description                                  |
|:-----------------------:|:------------------------------------------------------------------------------:|
|       datasets.py       |                       Pytorch Dataset for loading images                       |
|       gen_test.py       |              Data generation for non-aug and aug train test split              |
|       run_test.py       |                                   Get metrics                                  |
| streaming_dataloader.py |            Contain IterableDataset for both Drone and Ship dataset             |
|         utils.py        | Loss functions and helper functions for transforming data, rle decoding etc.)  |
|       ship_main.py      |                       Main script for ship model training                      |
|      drone_main.py      |                      Main script for drone model training                      |
|        notebooks/       |                       Misc jupyter notebooks for data viz                      |

## Usage

1. To generate cropped 256x256 test patches (Augmented and Non-Augmented)
```python 
python3 gen_test.py --input_img_ids INPUT_IMG_IDS_CSV --input_img_dir INPUT_IMG_DIR
    --imsave_dir IMSAVE_DIR --mask_imsave_dir MASK_IMSAVE_DIR
    --gpu 0
```

2. Model training

```python
# For Airbus Ship segmentation dataset
python3 ship_main.py

# For Drone segmentation dataset
python3 drone_main.py
```

3. Obtain test metrics (Augmented and Non-Augmented)

```python 
python3 run_test.py --input_fn INPUT_DIR --input_aug_fn INPUT_AUG_DIR
    --model_fn MODEL_FN --model_aug_fn AUG_MODEL_FN --gpu 0
```

