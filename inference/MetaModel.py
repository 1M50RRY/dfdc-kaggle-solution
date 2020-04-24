import torch
import torch.nn as nn

class MetaModel(nn.Module):
    def __init__(self, models=None, device='cuda:0', extended=False):
        super(MetaModel, self).__init__()
        
        self.extended = extended
        self.device = device
        self.models = models
        self.len = len(models)
        
        if self.extended:
            self.bn = nn.BatchNorm1d(self.len)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(self.len, 1)
        
    def forward(self, x):
        x = torch.cat(tuple(x), dim=1)
        
        if self.extended:
            x = self.bn(x)
            x = self.relu(x)
            #x = self.dropout(x)
            
        x = self.fc(x)
        
        return x