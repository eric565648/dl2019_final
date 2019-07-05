import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image

def getData(mode , Datafolder):
    if mode == 'train':
        
        img = pd.read_csv(Datafolder + '_train_img.csv')
        label = pd.read_csv(Datafolder + '_train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(Datafolder + '_test_img.csv')
        label = pd.read_csv(Datafolder + '_test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class Loader(Data.Dataset):
    def __init__(self, mode, Datafolder):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_name, self.label = getData(mode,Datafolder)
        self.mode = mode
        self.Datafolder = Datafolder
        print("> Found %d images..." % (len(self.img_name)))
        

    def __len__(self):
        #------------return the size of dataset
        return len(self.img_name)

    def __getitem__(self, index):
        #-------------Get the image path from 'self.img_name' and load it.
        root = './'+ self.Datafolder +'/'          
        path = root + self.img_name[index] + '.png'
        img = Image.open(path)
        img_as_img = img.resize(( 160, 120),Image.ANTIALIAS)
        
        #-------------Get the ground truth label from self.label"""
        label = torch.from_numpy(self.label)[index]
        
        #-------------Transform the .jpeg rgb images
        if self.mode == 'train':
            transform1 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-45,45), resample=False, expand=False, center=None),
                transforms.ColorJitter(contrast=(0,1)),
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                ]
            )
        else:
            transform1 = transforms.Compose([
                transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
                ]
            )
        img_trans = transform1(img_as_img)
                
        #-------------Return processed image and label
        return img_trans, label
