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

## Usage

1. To generate cropped 256x256 test patches (Augmented and Non-Augmented)
```python 
python3 gen_test.py
```

2. Model training

```python 
python3 main.py
```
3. Obtain test metrics (Augmented and Non-Augmented)

```python 
python3 run_test.py
```

