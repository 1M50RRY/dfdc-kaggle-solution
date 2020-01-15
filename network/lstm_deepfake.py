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

class LSTMDF(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, dropout=0.0):
        super(LSTMDF, self).__init__()

        self.model = return_pytorch04_xception()
        self.lstm = nn.LSTM(1000, 512, 2, dropout=dropout, batch_first = True)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.fc(x)
        
        return x

