<h2 align="center"> Person re-ID with contextual information </h2>

---

<p align="center"> For original repository and documentation refer to <a href="https://github.com/layumi/Person_reID_baseline_pytorch"> Person reID baseline pytorch </a>. <p>

![Python3.10+](https://img.shields.io/badge/python-3.10+-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## What is re-ID with contextual information?

Person re-identification (re-ID) is a computer vision task of detecting the same individual across different
frames or video tracklets. Particularly noteworthy is the challenge of cross-camera re-ID, the process of
recognizing the same individual in frames captured by different cameras.

The standard approach involves a convolutional neural network (CNN) that creates embedding feature vector
for a query frame and then compares it with a matrix of all gallery images using some distance metric, such
as cosine distance. The closer the query vector the higher the probability that a query and a gallery frames
contain the same person. ResNet50 is used as a CNN architecture.

However, this method tends to overlook valuable contextual information, which extends beyond mere pixel-
level image representation. Contextual data may include a range of factors, such as time, direction of
movement, or the relative location of cameras. This type of information, when available, can enhance the
re-ID process significantly.

## Setup

Python 3.10+ is required. First, create a virtual environment
```bash
python3 -m venv .venv <-- creates an environment within /.venv directory 
```
Activate it.
```bash
source .venv/bin/activate
```
(Optional) upgrade pip
```bash
python3 -m pip install --upgrade pip
```
Install dependencies.
```bash
python3 -m pip install -r requirements.txt
```

## Usage

This repository has several functionalities.

1. [Train](#model-training) a re-ID ResNet50 model with any dataset in a compatible format (otherwise refer to [Annot documentation](https://github.com/A-NGJ/Person_reID_baseline_pytorch/blob/master/annot/Readme.md)).
2. [Evaluate](#model-evaluation) trained model for Rank@1, Rank@5, Rank@10, and mAP. ST-reid augmentation can be used if data contains necessary information. That is absolute timestamp or sequence number with constant offest accross cameras on top of standard re-ID information.
3. [Prepare](https://github.com/A-NGJ/Person_reID_baseline_pytorch/blob/master/annot/Readme.md) dataset in expocted format.
4. (Experimental feature) [Cluster](#labels-clusering-using-pre-trained-model) people labels using pre-trained re-ID model. More precisely, feature vector representations of input images.

### Configuration file

Scripts `test.py`, `train.py`, and `predict.py` share one configuration file. It contains following configurable values. Note that not all the information is used in every script.

- **batchsize: int**  
    Default: 32  
- **data_dir: str**  
    Absolute path to data directory. Data has to be in the expected format.
- **debug_test: bool**  
    Whether to save top 10 retrieved images for each query image. Images are saved to `debug` folder within `data_dir` directory. Default: false.
- **droprate: float**  
    Dropout rate used in model training. Default: 0.5
- **epochs: int**  
    Number of training epochs. Default: 60.
- **gpu_ids: List[int]**  
    List of used gpus ids. Default: [0]
- **height: int**  
    Image height after resizing. Default: 256
- **linear_num: int**  
    Length of feature vector in the next-to-last layer. Default: 512
- **lr: float**  
    Training learning rate.
- **model_dir: str**  
    Directory to load/save model. Default: `/model`.
- **model_name: str**  
    Name to load/save model. Default: `/resnet`.
    If model with a given name already exists it gets overwritten.
- **ms: List[float]**  
    Multiple scale. Used in evaluation. Default: [1].
- **nclasses: int**  
    Number of classes to predict. Should be equal to number of distinct labels in a training dataset. It gets automatically propated to testing script from a saved snapshot of a config used for training.
- **no_cache: bool**  
    Whether to overwrite cached feature tensors. Used to speed up evaluation process if run multiple times on the same dataset. Default: false.
- **patience: int**  
    Early stopping patience. Used in training. Default: 5.
- **results_path: str**  
    Path to catched results that contain feature tensors. Used in evaluation. Default: `cache/results.mat`.
- **st_reid: bool**  
    Whether to use ST-reid. ST-reid is a method that augments contextual information to the model predictions.
- **step: int**
    If `test_dataloader` is set to `context_video`, use every $n_{th}$ frame where $n = \text{step}$. Default: 1.
- **stride: int**  
    Convolutional layer stride. Default: 2.
- **test_dataloader: str**  
    Choose from {reid, context_video}. Which dataloader to use for evaluation. `reid` should be used to image datasets, `context_video` for video datasets where timestamp information is available.
- **verbose: bool**  
    Verbose model. Default: false.
- **warm_epoch: int**  
    Number of warm-up epochs. Default: 0.
- **weight_decay: float**
    Also known as L2 regularization or Ridge regression. It is applied to the weights of the neural network to help prevent overfitting and improve generalization. Default: 0.0005.
- **width: int**
    Image width after resizing. Default: 128.

### Dataset format

The data should be stored in the following format. You can use `annot/annot.py` to parse your data. For more details refer to [Annot documentation](https://github.com/A-NGJ/Person_reID_baseline_pytorch/blob/master/annot/Readme.md).

```bash
root
├── gallery
│   ├── 0000
│   │   ├── 0000_c00_0001.jpg
│   │   ├── 0000_c00_0002.jpg
│   │   └── 0000_c01_0001.jpg
├── query
│   ├── 0000
│   │   ├── 0000_c00_0001.jpg
├── train
│   ...
├── val
```

Each of directories located in `root`, i.e. `gallery`, `query`, `train`, and `val` should contain one subdirectory for each individual person. Each subdirectory contains frames for that person in the `<ID>_c<camera_id>_<timestamp|sequence_no>` format. `ID` is zero-padded up to 4 digits, `<camera_id>` zero-padded to two-digits and `<timestamp|sequence_no> must be always the last sequence of digits before file extension.

Training requires `train` and `val` directories. Optionally, `val` can be omitted, but you won't get validation loss and accuracy and early stopping won't work (it relies on the validation loss).

Testing requires `gallery` and `query`. Each query sample ID should be present in gallery. A gallery can contain more IDs, e.g. noise or images of irrelevant people, to mislead a model.

> IMPORTANT. Training directories must contain different person IDs than those in testing directories. Otherwise evaluation is biased, because a model is evaluated on people that it has already seen. In other words, it is impossible to tell if it generalizes well.

### Model training

Use `train.py` to train a model. Command line arguments overwrite JSON configuration (only those that were specified, everything else is taken from config). Therefore all CLI arguments are optional. For default values refer to [configuration description](#configuration-file).

```bash
python train.py
[-f <config_file>; Default config.json] Configuration file to use in the JSON format.
[--gpu-ids] [--model-name] [--data-dir] [--batchsize] [--lr] [--weight-decay] [--epochs] [--linear-num] [--stride] [--droprate] [--warm-epoch] [--patience] [--verbose]

- python train.py --batchsize 128 <-- use default parameters, but overwrite batch size to 128.
```

### Model evaluation

Use `test.py` to evaluate a model. Command line arguments overwrite JSON configuration (only those that were specified, everything else is taken from config). Therefore all CLI arguments are optional. For default values refer to [configuration description](#configuration-file).

```bash
python test.py
[-f <config_file>; Default config.json] Configuration file to use in the JSON format.
[--gpu-ids] [--model-name] [--data-dir] [--batchsize] [--linear-num] [--dataloader] [--step] [--st-reid] [--no-cache] [--debug]

python test.py --st-reid --no-cache --debug <-- set flags to use ST-reid, overwrite cache and debug evaluation predictions.
```

### Labels clusering using pre-trained model.

Use `predict.py` to cluster unknown people labels. This is an experimental feature.

```bash
python predict.py
[-f <config_file>; Default config.json] Configuration file to use in the JSON format.
[--model-name] [--data-dir] [--batch-size] [--dataloader]
```