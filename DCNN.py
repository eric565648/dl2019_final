import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import Net
import Loader
parser = argparse.ArgumentParser(description='dl2019 project')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--data-folder', default='set00', required=True,
					help='Data Folde (default: set00)')
		
args = parser.parse_args()

LR = args.lr
BATCH_SIZE = args.batch_size 
EPOCH = args.epochs
print("=========parameter========")
print("learning rate =",LR)
print("batch size =" , BATCH_SIZE)
print("epochs =" ,EPOCH)
print("data folder =",args.data_folder)
print("==========================")
TRAIN_LOSS=[]

#DATA_FOLDER = 'set_00'
DATA_FOLDER = args.data_folder

# Dataloader
LOADER_PATH = './'+ DATA_FOLDER +'/'
train_data=Loader.Loader(LOADER_PATH ,'train',DATA_FOLDER)
test_data=Loader.Loader('./'+ DATA_FOLDER +'/','test',DATA_FOLDER)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
print (train_loader)
print (test_loader)

def train_DeepConv(epoch):
    model_DeepConvNet.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.to(device=device, dtype=torch.float), target.to(device)

        optimizer.zero_grad()
        output = model_DeepConvNet(data)
        target = target.float()
        loss_d = Loss(output[0], target[0])
        loss_phi = Loss(output[1], target[1])
        loss = loss_d + loss_phi
        loss.backward()
        optimizer.step()
    TRAIN_LOSS.append(round((loss.data).cpu().numpy().tolist(),6))
   
    print('Train Epoch: {} \t Loss_d: {} Loss_phi:{}'.format(
            epoch, loss_d.data , loss_phi.data))

def test_DeepConv(epoch):
    model_DeepConvNet.eval()
    test_loss_d = 0.0
    test_loss_phi = 0.0
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = data.to(device=device, dtype=torch.float), target.to(device)
        with torch.no_grad():
            output = model_DeepConvNet(data)
        target = target.float()
        test_loss_d = Loss(output[0], target[0])
        test_loss_phi = Loss(output[1], target[1])
        
    print('---------------Test set: Average loss d:{}\t loss phi:{}-------------------\n'
          .format(test_loss_d,test_loss_phi))

model_DeepConvNet = Net.DeepConvNet().cuda(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
    model_DeepConvNet.to(device) 
#------------define optimizer/loss function
Loss = nn.MSELoss(reduction='mean')    
optimizer = torch.optim.SGD(model_DeepConvNet.parameters(), lr=LR, momentum=0.9, 
                            dampening=0, weight_decay=0.0005, nesterov=False)

print("======Start training======")

#---------------Train-----------------
for epoch in range(0, EPOCH):
    train_DeepConv(epoch)
    test_DeepConv(epoch)
    if ( epoch % 100 == 0 ):
        PATH = 'model.pkl'
        if( epoch % 500 == 0 ):
        	PATH = str(epoch)+'model.pkl'	
        torch.save(model_DeepConvNet.state_dict(), PATH)
