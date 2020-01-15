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
from random import randint

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [f.resize([int(d * self.resize) for d in f.size]) for f in frames]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        #print(probs)
        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            if float(probs[box_ind][0]) > 0.99:
                for box in boxes[box_ind]:
                    faces.append(frame.crop(box))
        
        return faces

def extract_faces(video_path, fast_mtcnn, transforms, limit=1, delimeter=1):
    frames = []
    #delimeter = randint(1, 20)
    try:
        v_cap = FileVideoStream(video_path).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if limit > v_len:
            limit = v_len

        for j in range(v_len):
            if len(frames) == limit:
                break

            if j % delimeter == 0:
                frame = v_cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)

        faces = fast_mtcnn(frames)
        v_cap.stop()
        
        return [transforms(face).numpy() for face in faces]
    except Exception as e:
        print(str(e))
        return [transforms(frame).numpy() for frame in frames]