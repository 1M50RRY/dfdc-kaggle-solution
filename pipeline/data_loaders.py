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
from imutils.video import FileVideoStream
import gc

'''Images ds loader'''

def load_img_dataset(data_path, batch_size, resize=256, normalize=torchvision.transforms.Normalize((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))):
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