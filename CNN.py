import torch
import torch.nn as nn
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,32,3,padding=1,stride=2)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,64,3,padding=1,stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(200704,100)
        self.linear2 = nn.Linear(100,6)
        self.ReLU = nn.ReLU()
        

    def forward(self, X):
        out = self.conv1(X)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.ReLU(out)
        out = self.conv3(out)
        out = self.ReLU(out)
        out = self.conv4(out)
        out = self.ReLU(out)
        out = self.flatten(out)
        out = self.ReLU(out)
        out = self.linear1(out)
        out = self.ReLU(out)
        out = self.linear2(out)
        out = self.ReLU(out)
        return out
        
    
