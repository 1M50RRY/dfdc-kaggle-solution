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
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, ImageCompression, Resize, RandomScale, RandomFog, RandomShadow, Downscale, JpegCompression, CenterCrop,
    RandomGamma, RandomContrast, Cutout, Lambda, Cutout, Posterize
)
from pipeline.helpers.balanced_batch_sampler import BalancedBatchSampler, make_weights_for_balanced_classes
from torchsampler import ImbalancedDatasetSampler

def strong_aug(p=1):
    return Compose([
        OneOf([JpegCompression(quality_lower=15, quality_upper=50, p=1), Downscale(scale_min=0.5, scale_max=0.9, p=1)], p=0.5),
        OneOf([IAAAdditiveGaussianNoise(p=1), GaussNoise(p=1)], p=0.12),
        #OneOf([CLAHE(clip_limit=2,p=1), IAASharpen(p=1), IAAEmboss(p=1)], p=0.3),
        RandomGamma(p=0.15),
        RandomBrightnessContrast(p=0.15),
        #OneOf([MotionBlur(blur_limit=5,p=1),MedianBlur(blur_limit=5, p=1),Blur(blur_limit=5, p=1)], p=0.2),
        #OpticalDistortion(p=0.25),
        #HueSaturationValue(hue_shift_limit=10, p=0.2),
        #RandomRotate90(p=0.2),
        HorizontalFlip(p=0.5),
        #ShiftScaleRotate(p=0.2),
        #Cutout(p=0.2),
        #RandomFog(p=0.2),
        #Posterize(p=0.2)
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

        return sample, int('\\fake\\' in path.lower())


def load_img_dataset(data_path, batch_size, resize=256, normalize=torchvision.transforms.Normalize((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))):
    train_dataset = ImageFolderAlbum(
        root=data_path,
        transform=Compose([
            Resize(260, 260),
            #CenterCrop(240, 240),
            strong_aug(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            #Resize(240, 240),
            ToTensor(),
            #Lambda(lambda img: img * 2.0 - 1.0),
            ])
    )
    weights = make_weights_for_balanced_classes(
        train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        sampler=ImbalancedDatasetSampler(train_dataset),
        #sampler=sampler,
        #sampler=BalancedBatchSampler(train_dataset),
        pin_memory=True,
        shuffle=False
    )
    return train_loader


def load_img_val_dataset(data_path, batch_size, resize=256, normalize=torchvision.transforms.Normalize((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))):
    train_dataset = ImageFolderAlbum(
        root=data_path,
        transform=Compose([
            Resize(260, 260),
            #CenterCrop(240, 240),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor(),
            #Lambda(lambda img: img * 2.0 - 1.0),
            ])
    )
    weights = make_weights_for_balanced_classes(
        train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        #sampler=ImbalancedDatasetSampler(train_dataset),
        sampler=sampler,
        #sampler=BalancedBatchSampler(train_dataset),
        pin_memory=True,
    )
    return train_loader
