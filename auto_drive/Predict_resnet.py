import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image

import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------------parameter---------------

# model_path = './Res18.pkl'
#------------------------------------------------------------------------------------parameter---------------

# class Loader(Data.Dataset):
#     def __init__(self):
#         """
#         Args:
#             root (string): Root path of the dataset.
#             self.img_name (string list): String list that store all image names.
#             self.label (int or float list): Numerical list that store all ground truth label values.
#         """
#
#         self.img_name = args.data_filename
#         print("> Found  images..%s" % (self.img_name))
#
#
#     def __len__(self):
#         #------------return the size of dataset
#         return len(self.img_name)
#
#     def __getitem__(self, index):
#         #-------------Get the image path from 'self.img_name' and load it.
#
#         img = Image.open(DATA_PATH)
#         box=(0,120,640,480)
#         img_as_img=img.crop(box)
#
#         #-------------Transform the .jpeg rgb images
#         transform1 = transforms.Compose([
#                 transforms.Resize(( 320, 180)),
#                 transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
#                 #transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
#
#                 ]
#             )
#         img_trans = transform1(img_as_img)
#
#         #-------------Return processed image and label
#         return img_trans
# # Dataloader
# predict_data = Loader()
# predict_loader = Data.DataLoader(dataset=predict_data, batch_size=1)

# def test_DeepConv():
#     Model.eval()
#
#     for batch_idx, data in enumerate(predict_loader):
#         if torch.cuda.is_available():
#             data = data.to(device=device, dtype=torch.float)
#         with torch.no_grad():
#             output = Model(data)
#     output_numpy=output.cpu().numpy().squeeze()
#
#     print('---------------predict  d:{}\t  phi:{}-------------------\n'
#            .format(output_numpy.data[0],output_numpy.data[1]))

# Model = models.resnet18(pretrained=True).cuda(0)
# Model.fc=nn.Linear(512 * 1, 2 ,bias=True)
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     Model.to(device)
#
# Model.load_state_dict(torch.load(model_path))
# Model.eval()

#---------------Predict-----------------
# print("======Perdict======")
# test_DeepConv()

class Predict_ResNet(object):
	"""docstring for Predict_ResNet."""
	def __init__(self):

		model_path = './Res18.pkl'

		if torch.cuda.is_available():
			self.Model = models.resnet18(pretrained=True).cuda(0)
		else:
			self.Model = models.resnet18(pretrained=True)

		self.Model.fc=nn.Linear(512 * 1, 2 ,bias=True)
		if torch.cuda.is_available():
		    device = torch.device('cuda')
		    self.Model.to(device)

		self.Model.load_state_dict(torch.load('./Res18.pkl'))
		sefl.Model.eval()

	def predict(self, img):

		# self.Model.eval()
		assert state.shape[0] == 3

        if torch.cuda.is_available():
            data = torch.FloatTensor(np.expand_dims(img, axis=0)).to(device)

        with torch.no_grad():
            output = self.Model(data)

	    output_numpy=output.cpu().numpy().squeeze()

	    print('---------------predict  d:{}\t  phi:{}-------------------\n'
	           .format(output_numpy.data[0],output_numpy.data[1]))

		return output_numpy
