import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import json
import cv2
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
import torchvision
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tqdm
from PIL import ImageFilter, Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from facenet_pytorch import MTCNN, InceptionResnetV1
import gc
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations import RandomCrop, Normalize
from albumentations.pytorch import ToTensor
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, ImageCompression, Resize, RandomScale, RandomFog, RandomShadow
)
from pipeline.helpers.balanced_batch_sampler import BalancedBatchSampler, make_weights_for_balanced_classes

def strong_aug(p=.5):
    return Compose([
        ImageCompression(quality_lower=20, quality_upper=80, p=0.3),
        OneOf([
            OneOf([IAAAdditiveGaussianNoise(),GaussNoise(),], p=0.3),
            #OneOf([CLAHE(clip_limit=2),IAASharpen(),IAAEmboss(),RandomBrightnessContrast(),], p=0.3),
            #OneOf([MotionBlur(p=1),MedianBlur(blur_limit=3, p=1),Blur(blur_limit=3, p=1),], p=0.2),
            HueSaturationValue(p=0.2),
            ], p=0.5),
        OneOf([ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, p=1),RandomScale(scale_limit=0.5, p=1)], p=0.3),
        #RandomRotate90(),
        HorizontalFlip(p=0.5),
        #OneOf([OpticalDistortion(p=0.3),GridDistortion(p=.1),IAAPiecewiseAffine(p=0.3),], p=0.2),
        Resize(224, 224),
    ], p=p)


class ImageFolderAlbum(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(image=sample)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, int('fake' in path.lower())

def load_img_dataset(data_path, batch_size, resize=256, normalize=torchvision.transforms.Normalize((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))):
    train_dataset = ImageFolderAlbum(
        root=data_path,
        transform=Compose([ 
            strong_aug(p=1),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor()])
    )
    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=5,
        drop_last=True,
        sampler=sampler,
        #pin_memory=True,
        shuffle=False
    )
    return train_loader


def load_img_val_dataset(data_path, batch_size, resize=256, normalize=torchvision.transforms.Normalize((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))):
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((resize, resize)),
                                                  torchvision.transforms.ToTensor(),
                                                  normalize])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True
    )
    return train_loader
