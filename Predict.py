import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image

import Net

parser = argparse.ArgumentParser(description='dl2019 project')
parser.add_argument('--data-folder', default='set00',
					help='Data Folder (default: set00)')
parser.add_argument('--data-filename', default='0_1_1_1.png',
					help='Data Folde (default: 0_1_1_1.png)')
		
args = parser.parse_args()
DATA_PATH = args.data_folder + '/' +args.data_filename

print("=========parameter========")
print("data path =",DATA_PATH)
print("==========================")

class Loader(Data.Dataset):
    def __init__(self):
        """
        Args:
            root (string): Root path of the dataset.
            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """

        self.img_name = args.data_filename
        print("> Found  images..%s" % (self.img_name))
        

    def __len__(self):
        #------------return the size of dataset
        return len(self.img_name)

    def __getitem__(self, index):
        #-------------Get the image path from 'self.img_name' and load it.
                  
        img = Image.open(DATA_PATH)
        img_as_img = img.resize(( 160, 120),Image.ANTIALIAS)
        
        
        #-------------Transform the .jpeg rgb images
        transform1 = transforms.Compose([
            transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )
        img_trans = transform1(img_as_img)
                
        #-------------Return processed image and label
        return img_trans
# Dataloader
predict_data = Loader()
predict_loader = Data.DataLoader(dataset=predict_data, batch_size=1)

def test_DeepConv():
    model_DeepConvNet.eval()
    
    for batch_idx, data in enumerate(predict_loader):
        if torch.cuda.is_available():
            data = data.to(device=device, dtype=torch.float)
        with torch.no_grad():
            output = model_DeepConvNet(data)
    output_numpy=output.cpu().numpy().squeeze()
        
    print('---------------predict  d:{}\t  phi:{}-------------------\n'
           .format(output_numpy.data[0],output_numpy.data[1]))


model_DeepConvNet = Net.DeepConvNet().cuda(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
    model_DeepConvNet.to(device) 


model_path = './model/500-tanh.pkl'
model_DeepConvNet.load_state_dict(torch.load(model_path))
model_DeepConvNet.eval()



#---------------Predict-----------------
print("======Perdict======")
test_DeepConv()

