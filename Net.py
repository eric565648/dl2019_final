import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
    
        #####---1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #####
        
        #####---2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=(2, 2) ,bias=True, groups=2)
        self.batchnorm2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #####
        
        #####---3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1), bias=True, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1), bias=True, groups=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3 , stride=2, padding=0)
        #####
        
        self.fc6 = nn.Linear(in_features=1536, out_features=4096, bias=True)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=256, bias=False)
        self.fc9 = nn.Linear(in_features=256, out_features=2, bias=True)
        
    def forward(self, x):
        
        ###-----1------
        out = self.conv1(x)
        #out = self.batchnorm1(out)
        out = F.relu(out)
        out = self.pool1(out)        
        out = self.norm1(out)
        #out = F.dropout(out, p=0.5)
        
        ###-----2------
        out = self.conv2(out)
        #out = self.batchnorm2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.norm2(out)
        #out = F.dropout(out, p=0.5)
        
        ###-----3------
        out = self.conv3(out)
        out = F.relu(out)
        ###-----4------
        out = self.conv4(out)
        out = F.relu(out)
        
        out = self.conv5(out)
        out = F.relu(out)
        out = self.pool5(out)
        
        out = out.view(out.size(0), -1) #flatten
        out = self.fc6(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.5)
        
        out = self.fc7(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.5)
      
        out = self.fc8(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.5)
        
        out = self.fc9(out)
        out = torch.sigmoid(out)
        
        
        return out
