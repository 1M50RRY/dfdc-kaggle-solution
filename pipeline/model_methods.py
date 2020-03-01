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

def make_preds(net, x_batch):
    result = []
    
    for i in x_batch:
        result.append(net(i.unsqueeze(0))[0].detach().cpu().numpy())
    #print(torch.FloatTensor(np.asarray(result)))
    return torch.FloatTensor(result)

'''Main methods'''

def validate_img(net, X_test, y_train, loss, metric, device, batch_size, 
             print_results=False, 
             show_results=False, 
             show_graphic=False, 
             inference=None,
             checkpoint=None,
             reverse=False
            ):
    val_loss = []
    val_metrics = []
    net.eval()
    dataloader_iterator = iter(X_test)
    with torch.no_grad():
        for batch_idx in tqdm.tqdm_notebook(range(len(X_test))):
            #print(batch_idx)
            try:
                X_batch, y_batch = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(X_test)
                X_batch, y_batch = next(dataloader_iterator)


            if reverse:
                for i in range(len(y_batch)):
                    y_batch[i] = 1 if y_batch[i] == 0 else 0

            y_batch = torch.FloatTensor(y_batch).to(device)

            if inference:
                #test_preds = net.inference(X_batch.to(device))
                test_preds = inference(net(X_batch.to(device)))
                #test_preds = nn.functional.sigmoid(net(X_batch.to(device)))
            else:
                test_preds = net(X_batch.to(device))


            test_loss_value = F.binary_cross_entropy_with_logits(test_preds, y_batch).item() * batch_size
            val_loss.append(test_loss_value)
            
            if print_results or show_results:
                for i in range(int(len(test_preds) * 0.1)):
                    if show_results:
                        img = X_batch[i]
                        plt.title(str(float(test_preds[i])) + ', ' + str(float(y_batch[i])) + ' | ' + filenames[i])
                        #plt.title(str(float(test_preds[i][0])) + ', ' + str(float(y_batch[i])))
                        plt.imshow(img.permute(1, 2, 0).numpy())
                        plt.show()
                        plt.pause(0.001)
                    if print_results:
                        print(test_preds[i], y_batch[i])
            
            if show_graphic:
                submission = []
                for i in range(len(test_preds)):
                    submission.append([test_preds[i][1].cpu()])
                    #submission.append([test_preds[i][0].cpu()])
                submission = pd.DataFrame(submission, columns=['label'])
                plt.hist(submission.label, 20)
                plt.show()

            
            metrics = metric(torch.sigmoid(test_preds), y_batch)
            val_metrics += metrics
               
        mean_metrics = np.asarray(val_metrics).mean()
        mean_loss = sum(val_loss) / (len(X_test) * batch_size)

        if checkpoint is not None and (mean_metrics >= checkpoint[1] or mean_loss <= checkpoint[0]):
            torch.save(net.state_dict(), net.__class__.__name__  + ' ' + str(mean_metrics) + ' ' + str(mean_loss) + '.pth')
            #torch.save(net, str(mean_metrics) + ' ' + str(mean_loss) + '.h5')
            
        print('Validation: metrics ', mean_metrics, 'loss ', mean_loss)

        gc.collect()
        
        return mean_metrics, mean_loss

def train_img(net, loss, optimizer, scheduler, X_train, X_test, y_train, metric, device, val_metric, face_extractor, epochs=100, batch_size=100, 
          del_net=False, 
          useInference=False, 
          inference=False, 
          useScheduler=True,
          checkpoint=None,
          reverse=False
         ):   
    test_metrics_history = []
    test_loss_history = []


    for epoch in tqdm.tqdm_notebook(range(epochs)):
        dataloader_iterator = iter(X_train)
        net.train()
		
        train_loss = []
        train_metrics = []
		
        for batch_idx in tqdm.tqdm_notebook(range(len(X_train))):   
            try:
                X_batch, y_batch = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(X_train)
                X_batch, y_batch = next(dataloader_iterator)

            if reverse:
                for i in range(len(y_batch)):
                    y_batch[i] = 1 if y_batch[i] == 0 else 0
                
            if len(y_batch) > 0 and len(X_batch) == batch_size:
                optimizer.zero_grad()
                
                if (len(X_batch) == len(y_batch)):

                    y_batch = torch.FloatTensor(y_batch).to(device)

                    if useInference:
                        preds = inference(net(X_batch.to(device)))
                        #preds = net.inference(X_batch.to(device))
                    else:
                        preds = net(X_batch.to(device))
                    
                    metrics = metric(torch.sigmoid(preds), y_batch)
                    train_metrics.append(metrics)

                    loss_value = F.binary_cross_entropy_with_logits(preds, y_batch)#loss(preds, y_batch)
                    loss_value.backward()

                    optimizer.step()

                    if useScheduler:
                        scheduler.step()
                    
                    train_loss.append(loss_value.item() * batch_size)
            else:
                print("Unable to make an epoch: " + str(len(filenames)) + " found in y_train. " + str(len(X_batch)) + " batch size.")

       # reverse2 = not reverse
        test_metric_value, test_loss_value = validate_img(net, X_test, y_train, loss, metric, device, 10, inference=inference, checkpoint=checkpoint)
									
        #validate_vid_bf(net, X_test, y_train, val_metric[0], val_metric[1], device, 2, face_extractor, print_results=False, reverse=reverse, checkpoint=checkpoint)
        #validate_img(net, X_test, y_train, loss, metric, device, 10, inference=nn.Sigmoid(), checkpoint=checkpoint)
											
        mean_metrics = np.asarray(train_metrics).mean()
        mean_loss = sum(train_loss) / (len(X_train) * batch_size)
        
        print('Train: metrics ', mean_metrics, 'loss ', mean_loss)
        print('Expected LB value:', test_loss_value + 0.08228, test_loss_value * 100 / 83)
        print('Epoch:', epoch+1)

        test_loss_history.append(test_loss_value)
        test_metrics_history.append(test_metric_value)

        
        
    if del_net:
        del net

    gc.collect()

    return test_metrics_history, test_loss_history


def validate_vid(net, X_test, y_train, loss, metric, device, batch_size, fast_mtcnn, transforms,
             print_results=False, 
             show_results=False, 
             show_graphic=False, 
             inference=None,
             checkpoint=None,
             delimeter=20,
             remove_noise=False
            ):

    val_loss = []
    val_metrics = []
    val_metrics_ll = []
    val_metrics_ll_y = []
    net.eval()

    with torch.no_grad():
        for filename in tqdm.tqdm_notebook(os.listdir(X_test)):
            try:
                video = X_test + '\\' + filename
                y = y_train[y_train.name == filename.split('.')[0]].label.values[0]

                X_batch = []#torch.FloatTensor(extract_faces_dlib(video, fast_mtcnn, transforms, limit=batch_size, delimeter=delimeter, remove_noise=remove_noise)).to(device)

                if len(X_batch) == 0:
                    X_batch = torch.FloatTensor(extract_faces(video, fast_mtcnn, transforms, limit=batch_size, delimeter=delimeter, remove_noise=remove_noise)).to(device)

                    if len(X_batch) == 0:
                        print("Undetected", y)
                        val_metrics.append(int(1 == y))
                        continue
                print(len(X_batch))

                y_batch = torch.tensor([y]*len(X_batch)).float().to(device)
                #print(len(X_batch), len(y_batch))
                if (len(X_batch) == len(y_batch)):
                    
                    test_preds = net(X_batch)

                    test_loss_value = loss(test_preds, y_batch)
                    #test_loss_value.backward()
                    val_loss.append(float(test_loss_value.mean()))

                    if inference:
                        #test_preds = net.inference(X_batch)
                        test_preds = inference(test_preds) #inference(net(X_batch))
                        #test_preds = nn.functional.sigmoid(net(X_batch))
                    
                    
                    
                    if show_graphic:
                        submission = []
                        for i in range(len(test_preds)):
                            #submission.append([test_preds[i][1].cpu()])
                            submission.append([test_preds[i][0].cpu()])
                        submission = pd.DataFrame(submission, columns=['label'])
                        plt.hist(submission.label, 20)
                        plt.show()

                    metrics = metric(test_preds, y_batch)
                    
                    val_metrics.append(round(metrics))
                    
                    if print_results or show_results:
                        for i in range(len(test_preds)):
                            if show_results:
                                img = X_batch[i]
                                plt.title(str(float(test_preds[i][1])) + ', ' + str(float(y_batch[i])) + ' | ' + filename)
                                #plt.title(str(float(test_preds[i][0])) + ', ' + str(float(y_batch[i])))
                                plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
                                plt.show()
                                plt.pause(0.001)
                        if print_results:
                            #print(torch.mean(test_preds[:, 1:]), y_batch[0])
                            print(metrics, y_batch[0])

            except Exception as e:
                print(str(e))

            gc.collect()
        #return val_metrics, val_metrics_y
        mean_metrics = np.asarray(val_metrics).mean()
        mean_loss = np.asarray(val_loss).mean()
        
        if checkpoint is not None and np.asarray(val_metrics).mean() >= checkpoint:
            torch.save(net.state_dict(), str(mean_metrics) + ' ' + str(mean_loss) + '.pth')
            #torch.save(net, str(mean_metrics) + ' ' + str(mean_loss) + '.p')
            
        print('Validation: metrics ', mean_metrics, 'loss ', mean_loss)

        return mean_metrics, mean_loss

def train_vid(net, loss, optimizer, scheduler, X_train, X_test, y_train, metric, device, fast_mtcnn, transforms, 
            epochs=100, 
            batch_size=100, 
            del_net=False, 
            useInference=False, 
            inference=None, 
            useScheduler=True,
            checkpoint=None,
            limit=10,
            delimeter=50,
            remove_noise=False
         ):   

    test_metrics_history = []
    test_loss_history = []
    x_train = os.listdir(X_train)

    for epoch in tqdm.tqdm_notebook(range(epochs)):
        order = np.random.permutation(len(x_train))

        if useScheduler:
            scheduler.step()

        net.train()
        
        train_loss = []
        train_metrics = []
        
        for start_index in tqdm.tqdm_notebook(range(0, len(x_train), batch_size)):
            try:
                optimizer.zero_grad()

                X_batch = []
                y_batch = []

                batch_indexes = order[start_index:start_index+batch_size]

                for i in batch_indexes:
                    filename = x_train[i]
                    video = X_train + '\\' + filename
                    
                    faces = extract_faces(video, fast_mtcnn, transforms, limit=batch_size, delimeter=delimeter, remove_noise=remove_noise)#extract_faces(video, fast_mtcnn, transforms, limit=limit, delimeter=delimeter, remove_noise=remove_noise)

                    X_batch += faces
                    y_batch += [y_train[y_train.name == filename.split('.')[0]].label.values[0]]*len(faces)
                
                X_batch = torch.FloatTensor(X_batch).to(device)
                y_batch = torch.tensor(y_batch).to(device)
                
                if (len(X_batch) == len(y_batch) and len(X_batch) > 0):
                    if useInference:
                        preds = inference(net(X_batch))
                        #preds = net.inference(X_batch)
                    else:
                        preds = net(X_batch)

                    loss_value = loss(preds, y_batch)
                    loss_value.backward()

                    optimizer.step()
                    
                    train_loss.append(float(loss_value.mean()))
                    #train_metrics_y.append(int(y_batch[0]))
                    #train_metrics.append(round(float(torch.mean(inference(preds)[:, 1:]))))
                    train_metrics.append(round(metric(inference(preds)[:, 1:], y_batch)))
                else:
                    print("Unable to make an epoch: " + filename + ". " + str(len(y_batch)) + " found in y_batch. " + str(len(X_batch)) + " batch size.")
            except Exception as e:
                print(str(e))
                    
        test_metric_value, test_loss_value = validate_vid(net, X_test, y_train, loss, metric, device, 10, fast_mtcnn, transforms,
                                            inference=inference, checkpoint=checkpoint, remove_noise=False)
                                            
        mean_metrics = np.asarray(train_metrics).mean()
        mean_loss = np.asarray(train_loss).mean()
        print('Train: metrics ', mean_metrics, 'loss ', mean_loss)
        
        test_loss_history.append(test_loss_value)
        test_metrics_history.append(test_metric_value)
    
    if del_net:
        del net

    gc.collect()
        
    return test_metrics_history, test_loss_history

# ---------------------------------------------------------------------------------------------------------------------

def validate_vid_bf(net, X_test, y_train, loss, metric, device, batch_size, face_extractor, fast_mtcnn=None, print_results=False, checkpoint=None, reverse=False):

    val_metrics = []
    val_metrics_ll = []
    val_metrics_ll_y = []
    net.eval()

    with torch.no_grad():
        for filename in tqdm.tqdm_notebook(os.listdir(X_test)):
            #try:
            video = X_test + '\\' + filename

            y = y_train[y_train.name == filename.split('.')[0]].label.values[0]

            if reverse:
                y = 1 if y == 0 else 0

            test_preds = predict_on_video(net, device, video, face_extractor, fast_mtcnn=fast_mtcnn, batch_size=batch_size)

            if type(test_preds) == float:
                if print_results:
                    print(0.5, y)
                val_metrics.append(int(1 == y))
                val_metrics_ll.append(0.5)
                val_metrics_ll_y.append(int(1 == y))
                continue

            metrics = metric(test_preds, y)
            val_metrics.append(round(metrics))

            val_metrics_ll.append(test_preds.mean())
            val_metrics_ll_y.append(y)
            
            if print_results:
                print(test_preds.mean(), y)

            #except Exception as e:
                #print(str(e))

            gc.collect()

        mean_metrics = loss(torch.FloatTensor(val_metrics_ll), torch.tensor(val_metrics_ll_y)) 
        mean_loss = np.asarray(val_metrics).mean()
        
        if checkpoint is not None and (mean_metrics <= checkpoint[0] or mean_loss >= checkpoint[1]):
            torch.save(net.state_dict(), str(mean_metrics) + ' ' + str(mean_loss) + '.pth')
            #torch.save(net, str(mean_metrics) + ' ' + str(mean_loss) + '.p')
            
        print('Validation: metrics ', mean_loss, 'loss: ', mean_metrics)

        return mean_metrics, mean_loss

def train_vid_bf(net, loss, optimizer, scheduler, X_train, X_test, y_train, metric, device, face_extractor, 
            frames=17,
            epochs=100, 
            batch_size=100, 
            del_net=False,   
            useScheduler=True,
            checkpoint=None
         ):   

    test_metrics_history = []
    test_loss_history = []
    x_train = os.listdir(X_train)

    for _ in tqdm.tqdm_notebook(range(epochs)):
        order = np.random.permutation(len(x_train))

        if useScheduler:
            scheduler.step()

        net.train()
        
        train_loss = []
        train_metrics = []
        
        for start_index in tqdm.tqdm_notebook(range(0, len(x_train), batch_size)):
            #try:
            optimizer.zero_grad()

            X_batch = []
            y_batch = []

            batch_indexes = order[start_index:start_index+batch_size]

            for i in batch_indexes:
                filename = x_train[i]
                video = X_train + '\\' + filename
                
                preds = predict_on_video(net, device, video, face_extractor, batch_size=frames, train=True)
                y = y_train[y_train.name == filename.split('.')[0]].label.values[0]

                if type(preds) == float:
                    continue

                X_batch.append(preds)
                y_batch += [float(y)]*len(preds)

                train_metrics.append(round(metric(preds.detach().cpu().numpy(), y)))
                
            X_batch = torch.cat(X_batch).to(device)
            y_batch = torch.tensor(y_batch).to(device)

            loss_value = loss(X_batch, y_batch)
            loss_value.backward()

            optimizer.step()
            
            train_loss.append(float(loss_value.mean()))
            #except Exception as e:
                #print(str(e))
                    
        test_metric_value = validate_vid_bf(net, X_test, y_train, loss, metric, device, 17, face_extractor, checkpoint=checkpoint) 
        mean_metrics = np.asarray(train_metrics).mean()
        mean_loss = np.asarray(train_loss).mean()
        print('Train: metrics ', mean_metrics, 'loss ', mean_loss)
        
        test_metrics_history.append(test_metric_value)
    
    if del_net:
        del net

    gc.collect()
        
    return test_metrics_history