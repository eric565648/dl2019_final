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
parser.add_argument('--data-folder', default='ele',
					help='Data Folder (default: ele)')
parser.add_argument('--data-filename', default='9.png',
					help='Data Folde (default: 9.png)')
		
args = parser.parse_args()

#================================================================parameter========
#DATA_PATH : input picture
DATA_PATH = args.data_folder + '/' +args.data_filename       

model_path = './DCNN_w.pkl'
#================================================================parameter========

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
                  
        img_as_img = Image.open(DATA_PATH)
        
        
        #-------------Transform the .jpeg rgb images
        transform1 = transforms.Compose([
#                 transforms.Resize(( 280, 210)),
                transforms.CenterCrop(480),
                transforms.Resize(( 101, 101)),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                #transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
             
                ]
            )
        img_trans = transform1(img_as_img)
                
        #-------------Return processed image and label
        return img_trans
# Dataloader
predict_data = Loader()
predict_loader = Data.DataLoader(dataset=predict_data, batch_size=1)

def test_DeepConv():
    Model.eval()
    
    for batch_idx, data in enumerate(predict_loader):
        if torch.cuda.is_available():
            data = data.to(device=device, dtype=torch.float)
    
        with torch.no_grad():
            output = Model(data)
    pred = output.data.max(1)[1]
    pred_out = pred.cpu().numpy()[0]
        
    print('---------------predict  W:{}\t-------------------\n'
           .format(pred_out))


Model = Net.DeepConvNet().cuda(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
    Model.to(device) 



Model.load_state_dict(torch.load(model_path))
Model.eval()



#---------------Predict-----------------
print("======Perdict======")
test_DeepConv()

