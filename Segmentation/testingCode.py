import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
#import csv
import numpy as np
#import random
#import torchvision.models as models
import PIL
from torchvision import transforms
import torch
#import torch.nn
#import torch.optim as optim
#import torch.nn as nn
#from torch.utils.data import DataLoader
#from tqdm import tqdm

#import argparse
#import pickle

import customDatasetSeg
from customDatasetSeg import CustomImageDataset


myDataset = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images', transform_data_seg_1 = customDatasetSeg.transform_data_seg_1, transform_toTensor = customDatasetSeg.transform_toTensor, transform_data_seg_2 = customDatasetSeg.transform_data_seg_2, list_im = None)
ex = myDataset.__getitem__(12)
ex_np = ex['image'].detach().numpy()
ex_np = np.moveaxis(ex_np, 0, -1)
for i in range(3):
    ex_np[:,:,i] = (ex_np[:,:,i] - np.min(ex_np[:,:,i]))/np.max(ex_np[:,:,i] - np.min(ex_np[:,:,i]))
print(np.max(ex_np), np.min(ex_np))
print(ex_np)


ex_np_ = ex['mask'].detach().numpy().transpose()
ex_np_ = (ex_np_ - np.min(ex_np_))/np.max(ex_np_)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ex_np)
plt.subplot(1,2,2)
plt.imshow(ex_np_)
plt.show()
plt.close()

exit()



transform_data = transforms.Compose([
    #transforms.Resize((256,256)),
    #transforms.RandomCrop(224),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
    #transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ColorJitter(0.4,0.4,0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[194.6979, 139.2626, 145.4852], std=[22.8551, 30.9032, 33.9032])
    ])   # mean and stddev have been previously calculated
    
transform_toTensor = transforms.Compose([
    transforms.ToTensor()
    ]) 

transform_data_seg_2 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
    transforms.RandomPerspective(distortion_scale=0.2),
    ])   # mean and stddev have been previously calculated

im = PIL.Image.open("../data/images/ISIC_0024306.jpg")
imTensor = transform_data(im)
print(imTensor)
imTensor2 = transform_data_seg_2(imTensor)
print(imTensor2)
print('imTensor2')
print(imTensor2.shape)
print(np.std(imTensor2.numpy(), axis=(1,2)))
print(imTensor2.numpy().shape)

imSeg = PIL.Image.open("../data/HAM10000_segmentations_lesion_tschandl/ISIC_0024306_segmentation.png")
imSegTensor = transform_toTensor(imSeg)

print('Image')
print(imTensor.shape)
print(np.std(imTensor.numpy(), axis=(1,2)))
print(imTensor.numpy().shape)

print('\nSeg Image')
print(imSegTensor.shape)
print(np.std(imSegTensor.numpy(), axis=(1,2)))
print(imSegTensor.numpy().shape)

conTensor = torch.cat((imTensor, imSegTensor), 0)
print('\nconTensor')
print(conTensor.shape)
print(np.std(conTensor.numpy(), axis=(1,2)))
print(conTensor.numpy().shape)

transformed_conTensor = transform_data_seg_2(conTensor)

print('\ntransformed_conTensor')
print(transformed_conTensor.shape)
print(np.std(transformed_conTensor.numpy(), axis=(1,2)))
print(transformed_conTensor.numpy().shape)

#---------------------------------------------------------------------------------------------------------#

print('-----------------------------------------------------------------------------------')
#print(imSegTensor[0,0])
#print(len(imSegTensor[0]))
#print(len(imSegTensor[0,15]))

#print(imSegTensor[0,0][0])



label = 'bcc'
dic = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
imSegTensor[imSegTensor==0] = 7
imSegTensor[imSegTensor==1] = dic[label]
print(imSegTensor)

exit()

#data2 = 
data1 = [[[1, 1, 1, 1], [1, 1, 1, 1]],[[1, 1, 1, 1], [1, 1, 1, 1]], [[1,1, 1, 1], [1, 1, 1, 1]]]
data2 = [[[2, 2, 2, 2], [2, 2, 2, 2]]]
x_data1 = torch.tensor(data1)
x_data2 = torch.tensor(data2)
print(x_data1)
print(x_data1.shape)

print(x_data2)
print(x_data2.shape)

con_x_data = torch.cat((x_data1, x_data2), 0)
print(con_x_data.shape)

'''
con_x_data[3][con_x_data[3]==2] = 7

print(con_x_data)
print('--------------')
print(con_x_data[3])
print(con_x_data[3] == x_data2)'''

split_con_x_data = torch.split(con_x_data, 3)[1]
print('split_con_x_data: ', split_con_x_data)


#print(imSegTensor)
#print(imSegTensor[0,15])


print('dic[label]: ', dic[label])



#for row in range(len(imSegTensor[0])): # iterates through rows
#    for col in range(len(imSegTensor[0,0])):
        