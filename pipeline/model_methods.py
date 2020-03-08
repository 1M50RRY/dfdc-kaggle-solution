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
import glob
from pipeline.image_extracting import extract_faces, extract_faces_dlib, predict_on_video

def validate_img(net, X_test, y_train, metric, device, batch_size, checkpoint=None):
    val_loss = []
    val_metrics = []
    net.eval()
    dataloader_iterator = iter(X_test)
    with torch.no_grad():
        for _ in tqdm.tqdm_notebook(range(len(X_test))):
            try:
                X_batch, y_batch, paths = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(X_test)
                X_batch, y_batch, paths = next(dataloader_iterator)

            y_batch = torch.FloatTensor(y_batch).to(device)
            test_preds = net(X_batch.to(device))


            test_loss_value = F.binary_cross_entropy_with_logits(test_preds, y_batch).item() * batch_size
            val_loss.append(test_loss_value)

            metrics = metric(torch.sigmoid(test_preds), y_batch)
            val_metrics += metrics
               
        mean_metrics = np.asarray(val_metrics).mean()
        mean_loss = sum(val_loss) / (len(X_test) * batch_size)

        if checkpoint is not None and mean_loss <= checkpoint:
            torch.save(net.state_dict(), net.__class__.__name__  + ' ' + str(mean_metrics) + ' ' + str(mean_loss) + '.pth')
            #torch.save(net, str(mean_metrics) + ' ' + str(mean_loss) + '.h5')
            
        print('Validation: metrics ', mean_metrics, 'loss ', mean_loss)

        gc.collect()
        
        return mean_metrics, mean_loss

def train_img(net, optimizer, scheduler, X_train, X_test, y_train, metric, device, val_metric, epochs=100, batch_size=100, useScheduler=True,checkpoint=None):   
    test_metrics_history = []
    test_loss_history = []


    for epoch in tqdm.tqdm_notebook(range(epochs)):
        dataloader_iterator = iter(X_train)
        net.train()
		
        train_loss = []
        train_metrics = []
		
        for batch_idx in tqdm.tqdm_notebook(range(len(X_train))):   
            try:
                X_batch, y_batch, paths = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(X_train)
                X_batch, y_batch, paths = next(dataloader_iterator)
                
            if len(y_batch) > 0 and len(X_batch) == batch_size:
                optimizer.zero_grad()
                
                if (len(X_batch) == len(y_batch)):

                    y_batch = torch.FloatTensor(y_batch).to(device)
                    preds = net(X_batch.to(device))
                    
                    metrics = metric(torch.sigmoid(preds), y_batch)
                    train_metrics.append(metrics)

                    loss_value = F.binary_cross_entropy_with_logits(preds, y_batch)#loss(preds, y_batch)
                    loss_value.backward()

                    optimizer.step()
                    
                    train_loss.append(loss_value.item() * batch_size)
            else:
                print("Unable to make an epoch")
                
        if useScheduler:
                        scheduler.step()

        test_metric_value, test_loss_value = validate_img(net, X_test, y_train, metric, device, 10, checkpoint=checkpoint)
						
        mean_metrics = np.asarray(train_metrics).mean()
        mean_loss = sum(train_loss) / (len(X_train) * batch_size)
        
        print('Train: metrics ', mean_metrics, 'loss ', mean_loss)
        print('Expected LB value:', test_loss_value + 0.08228, test_loss_value * 100 / 83)
        print('Epoch:', epoch+1)

        test_loss_history.append(test_loss_value)
        test_metrics_history.append(test_metric_value)

    gc.collect()

    return test_metrics_history, test_loss_history