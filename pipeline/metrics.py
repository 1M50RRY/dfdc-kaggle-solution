import torch
import numpy as np
import pandas as pd

def accuracy(y, y0):
    return (y.argmax(dim=1) == y0).float().mean().data.cpu()

def accuracy_b(y, y0):
    y = y.detach().cpu().numpy()
    y0 = y0.detach().cpu().numpy()
    return [1 if round(float(y_i[0])) == float(y0_i[0]) else 0 for y_i, y0_i in zip(y, y0)]
    #return len([1 for y_i, y0_i in zip(y, y0) if round(y_i[0]) == y0_i]) / len(y)

def accuracy_b_mean(y, y0):
    y = y.detach().cpu().numpy()
    y0 = y0.detach().cpu().numpy()
    return 1 if round(y.mean()) == y0[0] else 0
    #return len([1 for y_i, y0_i in zip(y, y0) if round(y_i[0]) == y0_i]) / len(y)

def accuracy_sigmoid(y, y0):
    return len([1 for y_i in y if round(y_i) == y0]) / len(y)

def accuracy_sigmoid_mean(y, y0):
    return 1 if round(y.mean()) == y0 else 0

def log_loss_sigmoid(y, y0):
    loss = (-1/len(y0)) * sum([y0[i] * torch.log(y[i]) + (1 - y0[i]) * torch.log(1 - y[i]) for i in range(len(y))])
    return float(loss)

def log_loss(y, y0):
    #print(y, y0)
    loss = (-1/len(y0)) * sum([y0[i] * torch.log(y[i][1]) + (1 - y0[i]) * torch.log(1 - y[i][1]) for i in range(len(y))])
    return loss

def log_loss_b(y, y0):
    loss = (-1/len(y0)) * sum([y0[i] * torch.log(y[i]) + (1 - y0[i]) * torch.log(1 - y[i]) for i in range(len(y))])
    return float(loss)