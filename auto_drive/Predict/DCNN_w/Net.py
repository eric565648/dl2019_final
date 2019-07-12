import torch
import torch.nn as nn
import torch.nn.functional as F



class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
    
        #####---1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #####
        
        #####---2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1 )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #####
        
        #####---3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #####
        
        #####---4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #####
        
         
        #####---5
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #####
        
        
        #####---6
        self.conv6 = nn.Conv2d(64, 64, kernel_size=4, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #####
        
        
        self.fc5 = nn.Linear(in_features=64, out_features=500, bias=True)
        self.fc6 = nn.Linear(in_features=500, out_features=15, bias=True)

        
    def forward(self, x):
        
        ###-----1------
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)      
        out = F.dropout(out, p=0.5)
        
        ###-----2------
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = F.dropout(out, p=0.5)
       
        ###-----3------
        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool3(out)
        out = F.dropout(out, p=0.5)
        
        ###-----4------
        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool4(out)
        
        
        ###-----5------
        out = self.conv5(out)
        out = F.relu(out)
        out = self.pool5(out)
        out = F.dropout(out, p=0.5)
        
        out = out.view(out.size(0), -1) #flatten
        out = self.fc5(out)
        out = self.fc6(out)
        #out = F.softmax(out)
        #out = torch.tanh(out)
        #out = F.softsign(out)
        
        
        return out