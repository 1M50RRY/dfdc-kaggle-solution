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
import dlib

from albumentations import (
    Resize
)

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()

        return faces    


def process_faces(faces, resnet, device, limit):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    faces = torch.cat(faces[:limit]).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = resnet(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    centroid = embeddings.mean(dim=0)
    x = (embeddings - centroid).norm(dim=1).cpu().detach().numpy()
    
    return x

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

def extract_faces(video_path, fast_mtcnn, transforms, resnet=None, remove_noise=False, limit=1, delimeter=1):
    frames = []
    #delimeter = randint(1, 20)
    try:
        v_cap = FileVideoStream(video_path).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):
            if len(frames) == limit:
                break

            if j % delimeter == 0:
                frame = v_cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
                frame = Image.fromarray(frame)
                frames.append(frame)

        faces = fast_mtcnn(frames)

        v_cap.stop()

        if resnet:

            faces = torch.FloatTensor(faces).gpu()
             # Generate facial feature vectors using a pretrained model
            embeddings = resnet(faces)

            # Calculate centroid for video and distance of each face's feature vector from centroid
            centroid = embeddings.mean(dim=0)
            x = (embeddings - centroid).norm(dim=1).cpu().numpy()

            return x

        else:
            faces = [transforms(Image.fromarray(cv2.fastNlMeansDenoisingColored(np.asarray(face),None,10,10,7,21))).numpy() for face in faces] if remove_noise else [transforms(face).numpy() for face in faces]
            #faces = [transforms(frame).numpy() for frame in frames]
            if len(faces) > 0:
                return faces
            else:
                return [transforms(frame).numpy() for frame in frames]

    except Exception as e:
        print(str(e))
        if resnet:
            return None
        else:
            return [transforms(frame).numpy() for frame in frames]


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, transforms, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = transforms
    preprocessed_image = preprocess(Image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def extract_faces_dlib(video_path, fast_mtcnn, transforms, resnet=None, remove_noise=False, limit=1, delimeter=1):
    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    frame_num = 0

    result = []

    while reader.isOpened():
        _, image = reader.read()
        if image is None or (frame_num >= limit and len(result) > 0):
            break
        
        if frame_num % delimeter != 0:
            continue
        else:
            frame_num += 1

        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            preprocessed_image = preprocess_image(cropped_face, transforms, True).detach().cpu().numpy()
            result.append(preprocessed_image[0])
            # ------------------------------------------------------------------
        # Show
        cv2.waitKey(33)     # About 30 fps
    
    return result

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def predict_on_video(model, device, video_path, face_extractor, fast_mtcnn=None, batch_size=17, input_size=224, train=False, normalize_transform=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    #resized_face = isotropically_resize_image(face, input_size)
                    #resized_face = make_square_image(resized_face)
                    resized_face = torchvision.transforms.Resize((224, 224))(Image.fromarray(face))
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        #print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                        pass
                    
                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            if n > 0:
                x = torch.tensor(x, device=device).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                if not train:
                    # Make a prediction, then take the average.
                    with torch.no_grad():
                        y_pred = model(x)
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        return y_pred[:n].detach().cpu().numpy()
                else:
                    return torch.sigmoid(model(x).squeeze())[:n]

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    if fast_mtcnn:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
            faces = extract_faces(video_path, fast_mtcnn, transforms, resnet=None, remove_noise=False, limit=17, delimeter=10)

            if len(faces) > 0:
                with torch.no_grad():
                    x = torch.FloatTensor(faces).to(device)
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred.detach().cpu().numpy()

        except:
            pass
        
    return 0.5



def crop_faces(video_path, face_extractor, batch_size=1, input_size=256):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    #resized_face = isotropically_resize_image(face, input_size)
                    #resized_face = make_square_image(resized_face)
                    resized_face = torchvision.transforms.Resize((input_size, input_size))(Image.fromarray(face))
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        #print("WARNING: have %d faces but batch size is %d" % (n, batch_size))
                        pass
                    
                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            if n > 0:
                return x[:n][0]

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
        
    return None