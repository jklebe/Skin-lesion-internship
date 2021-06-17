from torch.utils.data import Dataset

import PIL
import csv
import torch

import PIL
from torchvision import transforms
import torch

#PATH = '../../data/'
#SEGPATH = '../../data/HAM10000_segmentations_lesion_tschandl/'

class CustomImageDataset(Dataset):
    

    def __init__(self, metaDataFile, path_to_images, path_to_masks, transform_data_seg_1 = None, transform_toTensor = None, transform_data_seg_2 = None, list_im = None):
        '''
        metaDataFile: path to labels
        path_to_images = path to images
        path_to_masks = path to masks
        transform_data_seg_1 = first transformations on image that do no apply to mask
        transform_toTensor = transforms mask to Tensor
        transform_data_seg_2 = transformations applied on image and mask simultaneously
        list_im = list with image names
        '''
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.labels_dir = metaDataFile
        #self.transform = transform
        self.transform_data_seg_1 = transform_data_seg_1
        self.transform_data_seg_2 = transform_data_seg_2
        self.transform_toTensor = transform_toTensor
        self.img_list = list_im
        if list_im == None:
            self.__fillList__()
            
    def __len__(self):
        return(len(self.img_list))
        
    def __fillList__(self):
        self.img_list = []
        with open(self.labels_dir, 'r' ) as csvfile:		
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')		
            next(reader)
            for row in reader:
                self.img_list.append(row[1]) 

    def __create4LayersImage__(self, image, imageSeg):
        ''' 
         returns concatenated tensor                                       
        '''
        #imTensor = transform_data_seg_1(image)
        #imSegTensoro = transform_toTensor(imageSeg)
        conTensor = torch.cat((image, imageSeg), 0)
        return conTensor
        
        
    def __getLabel__(self, imageName): 
        '''enter the name of the image and get the corresponding label'''
        dic = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
        with open(self.labels_dir, 'r' ) as csvfile:		
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')		
            next(reader)
            for row in reader:
                if imageName == row[1]:
                    return dic[row[2]]
    
   
    def __getitem__(self, idx):
        '''
        input: index (idx)
        output: {"image": image, "mask":mask, "name": name of image}
        '''
        im = PIL.Image.open(self.path_to_images + '/{}.jpg'.format(self.img_list[idx]))
        imSeg = PIL.Image.open(self.path_to_masks + '/{}_segmentation.png'.format(self.img_list[idx]))
        label = self.__getLabel__(self.img_list[idx])
        dic = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
        
        if self.transform_data_seg_1:
            image = self.transform_data_seg_1(im)
            
        if transform_toTensor:
            imageSeg = self.transform_toTensor(imSeg)
            
        # concatenate image and mask in order to perform final transformations
        conTensor = self.__create4LayersImage__(image, imageSeg)
        
        if self.transform_data_seg_2:
            conTensor = self.transform_data_seg_2(conTensor)
            
        # split image and mask 
        image = torch.split(conTensor, 3)[0]
        mask = torch.split(conTensor, 3)[1]
        
        # change mask values
        # black 0 -> 7
        # white 1 -> dic[label]
        mask[mask==0] = 7
        mask[mask!=7] = label 
        mask = mask.type(torch.long)
        
            
        sample = {"image": image, "mask":mask, "name": self.img_list[idx]}
        return sample



'''
Transforms images in regard of data augementation procedure.
Since ColorJitter and Normalization do not apply to mask transformation, these transformations
are applied solely on the image.
'''
transform_data_seg_1 = transforms.Compose([
    transforms.ColorJitter(0.4,0.4,0.4),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[194.6979, 139.2626, 145.4852], std=[22.8551, 30.9032, 33.9032])
    transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
    ])
    

transform_toTensor = transforms.ToTensor()
    
'''
Transforms image and mask simultaneously in regard of data augmentation procedure.
'''
transform_data_seg_2 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
    transforms.RandomPerspective(distortion_scale=0.2),
    ])
   

'''
Transform created for validation set.
'''   
transform_data_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
    ])


transform_data_seg_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[194.6979, 139.2626, 145.4852], std=[22.8551, 30.9032, 33.9032])
    transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
    ])


def transform_pg(level = 0):
    """
    transform for progressive growing
    level 0 :   7x  7
    level 1 :  14x 14
    level 2 :  28x 28
    level 3 :  56x 56
    level 4 : 112x112
    level 5 : 224x224
    """

    if level > 5:
        raise Exception()
    transform_data = transforms.Compose([
        transforms.Resize((2**(level + 3), 2**(level + 3))),
        transforms.RandomCrop(7 * 2 ** level),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        transforms.RandomPerspective(distortion_scale=0.2),
    ])

    return transform_data

