import os, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imutils.video import FileVideoStream 
from torchvision.transforms import Normalize

def disable_grad(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    return model
        

def weight_preds(preds, weights):
    final_preds = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if len(final_preds) != len(preds[i]):
                final_preds.append(preds[i][j] * weights[i])
            else:
                final_preds[j] += preds[i][j] * weights[i]
                
    return torch.FloatTensor(final_preds)


def predict_on_video(face_extractor, normalize_transform, stack_models, models, meta_models, weigths, video_path, batch_size, input_size, device):
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
                    resized_face = cv2.resize(face, (input_size, input_size))
                    
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            del faces

            if n > 0:
                x = torch.tensor(x, device=device).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction
                with torch.no_grad():
                    y_pred = 0
                    stacked_preds = []
                    preds = []
                    
                    for i in range(len(stack_models)):
                        stacked_preds.append(stack_models[i](x).squeeze()[:n].unsqueeze(dim=1))
                    
                    for i in range(len(models)):
                        preds.append(models[i](stacked_preds[meta_models[i][1]]))
                
                    del x, stacked_preds
                    
                    y_pred = torch.sigmoid(weight_preds(preds, weigths)).mean().item()
                    
                    del preds
                    
                    return y_pred

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
    
    
    return 0.5

