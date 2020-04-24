import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import cv2
import torchvision
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from albumentations import RandomCrop, Normalize
from albumentations.pytorch import ToTensor
from albumentations import (
    HorizontalFlip, Blur, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightness, OneOf, Compose, 
    ImageCompression, Resize, RandomScale, Downscale, JpegCompression, CenterCrop, GaussianBlur
)
from pipeline.helpers.balanced_batch_sampler import BalancedBatchSampler, make_weights_for_balanced_classes
from pipeline.helpers.imbalanced_batch_sampler import ImbalancedDatasetSampler

def strong_aug(p=1):
    return Compose([
        OneOf([JpegCompression(quality_lower=15, quality_upper=40, p=1), Downscale(scale_min=0.5, scale_max=0.9, p=1)], p=0.5),
        #OneOf([IAAAdditiveGaussianNoise(p=1), GaussNoise(p=1)], p=0.15),    
        RandomBrightness(p=0.15),
        #OneOf([MotionBlur(blur_limit=5,p=1),MedianBlur(blur_limit=5, p=1),Blur(blur_limit=5, p=1)], p=0.15),
        HorizontalFlip(p=0.5),
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

        return sample, torch.FloatTensor([float('\\fake\\' in path.lower())]), path


def load_img_dataset(data_path, batch_size, resize=None, crop=None, num_samples=None):
    transform = Compose([
            Resize(256, 256),
            strong_aug(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor(),
        ])

    if resize is not None:
        transform = Compose([
            Resize(resize, resize),
            strong_aug(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor(),
        ])

    if crop is not None:
        transform = Compose([
            Resize(256, 256),
            RandomCrop(crop, crop),
            strong_aug(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor(),
        ])
        
    train_dataset = ImageFolderAlbum(
        root=data_path,
        transform=transform
    )

    sampler = None

    if num_samples is not None:
        sampler = ImbalancedDatasetSampler(train_dataset, num_samples=num_samples)
    else:
        sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        drop_last=True,
        sampler=sampler,
        pin_memory=True,
        shuffle=False
    )
    return train_loader


def load_img_val_dataset(data_path, batch_size, resize=256):
    train_dataset = ImageFolderAlbum(
        root=data_path,
        transform=Compose([
            Resize(resize, resize),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensor(),
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
        num_workers=1,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        drop_last=False
    )
    return train_loader
