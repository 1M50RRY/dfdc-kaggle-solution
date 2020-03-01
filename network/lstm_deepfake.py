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
from network.models import return_pytorch04_xception
from efficientnet_pytorch import EfficientNet

class LSTMDF(nn.Module):
    def __init__(self, cnn, in_features, dropout=0.0):
        super(LSTMDF, self).__init__()

        self.dropout = dropout
        self.cnn = cnn
        self.lstm = nn.LSTM(in_features, 256, 2, dropout=self.dropout, batch_first = True)
        self.dp = nn.Dropout(self.dropout)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):

        batch_size, C, H, W = x.size()
        x = x.view(batch_size, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, 1, -1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.fc(self.dp(x))
        
        return x

