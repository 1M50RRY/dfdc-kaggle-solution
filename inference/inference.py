import os, sys, time
import cv2
import numpy as np
import pandas as pd
import random
from random import randint
from PIL import ImageFilter, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from imutils.video import FileVideoStream 
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import Normalize

from helpers.weigths_cfg import raw_data_stack, meta_models
from helpers.functions import disable_grad, weight_preds, predict_on_video
from helpers.blazeface import BlazeFace
from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

from MetaModel import MetaModel

if len(sys.argv) != 3:
    print("Usage: ./inference.py name_of_video num_frames")
    exit(0)


''' Initialize variables and helpers '''

WEIGTHS_PATH = "./pretrained/"
WEIGTHS_EXT = '.pth'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Using", device)

input_size = 256
frames_per_video = int(sys.argv[2])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

facedet = BlazeFace().to(device)
facedet.load_weights("./helpers/blazeface.pth")
facedet.load_anchors("./helpers/anchors.npy")
_ = facedet.train(False)

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)


''' Load and initialize models '''

models = []
weigths = []
stack_models = []

for raw_model in raw_data_stack:
    checkpoint = torch.load( WEIGTHS_PATH + raw_model[0] + WEIGTHS_EXT, map_location=device)
    
    if '-' in raw_model[1]:
        model = EfficientNet.from_name(raw_model[1])
        model._fc = nn.Linear(model._fc.in_features, 1)
    else:
        model = timm.create_model(raw_model[1], pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    
    model.load_state_dict(checkpoint)
    _ = model.eval()
    _ = disable_grad(model)
    model = model.to(device)
    stack_models.append(model)

    del checkpoint, model
    
for meta_raw in meta_models:

    checkpoint = torch.load(WEIGTHS_PATH + meta_raw[0] + WEIGTHS_EXT, map_location=device)
    
    model = MetaModel(models=raw_data_stack[meta_raw[1]], extended=meta_raw[2]).to(device)
    
    model.load_state_dict(checkpoint)
    _ = model.eval()
    _ = disable_grad(model)
    model.to(device)
    models.append(model)
    weigths.append(meta_raw[3])

    del model, checkpoint
    
total = sum([1-score for score in weigths])
weigths = [(1-score) / total for score in weigths]

print(predict_on_video(face_extractor, normalize_transform, stack_models, models, meta_models, weigths, sys.argv[1], frames_per_video, input_size, device))