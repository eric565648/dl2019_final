import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        
        #####---1
        self.conv1 = nn.Conv2d(3, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 5) ,bias=False)
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 5) ,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        #####
        
        #####---2
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 5) ,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        #####
        
        #####---3
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 5) ,bias=False)
        self.batchnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        #####
        
        #####---4
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 5) ,bias=False)
        self.batchnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        #####
       
        #classify
        self.fc1 = nn.Linear(in_features=1084600, out_features=2, bias=True)

        
    def forward(self, x):
        
        ###-----1------
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.batchnorm1(out)
            #Activation Func.---------------------------------
        out = F.relu(out)   
        
        out = self.maxpool1(out)
        out = F.dropout(out, p=0.5)
        
        ###-----2------
        out = self.conv3(out)
        out = self.batchnorm2(out)
            #Activation Func.---------------------------------
        out = F.relu(out)   
        
        out = self.maxpool2(out)
        out = F.dropout(out, p=0.5)
        
        ###-----3------
        out = self.conv4(out)
        out = self.batchnorm3(out)
            #Activation Func.---------------------------------
        out = F.relu(out)   
        
        out = self.maxpool2(out)
        out = F.dropout(out, p=0.5)
        
        ###-----4------
        out = self.conv5(out)
        out = self.batchnorm4(out)
            #Activation Func.---------------------------------
        out = F.relu(out)   
        
        out = self.maxpool2(out)
        out = F.dropout(out, p=0.5)
        #print(out.size())
        ###-----
        out = out.view(out.size(0), -1) #flatten
        #print(out.size())
        out = self.fc1(out)
        out = F.softsign(out)
        
        return out