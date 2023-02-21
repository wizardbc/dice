# DICE: Leveraging Sparsification for Out-of-Distribution Detection

This is the source code for ECCV 2022 paper [DICE: Leveraging Sparsification for Out-of-Distribution Detection](https://arxiv.org/abs/2111.09805)
by Yiyou Sun and Yixuan Li.

In this work, we reveal important insights that reliance on unimportant weights and units can directly attribute to the brittleness of OOD detection. To mitigate the issue, we propose a sparsification-based OOD detection framework termed DICE. Our key idea is to rank weights based on a measure of contribution, and selectively use the most salient weights to derive the output for OOD detection

## Usage

### 1. Dataset Preparation for Large-scale Experiment 

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/ILSVRC-2012/train` and  `./datasets/ILSVRC-2012/val`, respectively.

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/`.

### 2. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset


We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd datasets/ood_datasets
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```
### 3. Pre-trained Model Preparation

For CIFAR, the model we used in the paper is already in the checkpoints folder. 

For ImageNet, the model we used in the paper is the pre-trained ResNet-50 provided by Pytorch. The download process
will start upon running.

## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)


## Demo
### 1. Demo code for Large-scale Experiment 

Run `./demo-imagenet.sh`.

### 2. Demo code for CIFAR Experiment 

Run `./demo-cifar.sh`.

```
ID: CIFAR-100
No Sparsity
 OOD detection method: energy
              FPR    AUROC  AUIN  
SVHN          87.45  81.86  86.31
LSUN          14.71  97.43  97.63
LSUN_resize   70.67  80.13  81.41
iSUN          74.55  78.95  81.75
dtd           84.29  71.01  76.50
places365     78.22  78.15  14.17
AVG           68.32  81.26  72.96

With Sparsity p=90
 OOD detection method: energy
              FPR    AUROC  AUIN  
SVHN          59.24  88.57  90.33
LSUN           0.91  99.74  99.74
LSUN_resize   51.64  89.32  90.29
iSUN          49.46  89.51  91.00
dtd           61.40  77.12  79.35
places365     80.27  77.47  14.04
AVG           50.49  86.96  77.46


ID: CIFAR-10
No Sparsity
 OOD detection method: energy
              FPR    AUROC  AUIN  
SVHN          40.58  93.99  95.33
LSUN           3.82  99.15  99.25
LSUN_resize    9.28  98.12  98.49
iSUN          10.06  98.07  98.60
dtd           56.29  86.42  89.87
places365     39.73  91.81  38.50
AVG           26.63  94.59  86.67

With Sparsity p=90
 OOD detection method: energy
              FPR    AUROC  AUIN  
SVHN          29.64  94.66  95.20
LSUN           0.38  99.90  99.89
LSUN_resize    4.42  99.03  99.11
iSUN           5.14  98.97  99.16
dtd           45.85  86.97  88.82
places365     45.06  90.15  33.33
AVG           21.75  94.95  85.92
```

## Citation

If you use our codebase, please cite our work:
```
@inproceedings{sun2022dice,
  title={DICE: Leveraging Sparsification for Out-of-Distribution Detection},
  author={Sun, Yiyou and Li, Yixuan},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
