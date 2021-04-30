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
        ''' path_to_images = path to images, metaDataFile: path to labels'''
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

    # Aenderung 2
    def __create4LayersImage__(self, image, imageSeg):
        ''' makes image and the image containing the segmentation (imageSeg) '''
        ''' to tensor, applies colorJitter to image and normalizes image     '''
        ''' adds a forth layer to the image which is the segmentation image  '''
        ''' return concatenated tensor                                       '''
        #imTensor = transform_data_seg_1(image)
        #imSegTensoro = transform_toTensor(imageSeg)
        conTensor = torch.cat((image, imageSeg), 0)
        return conTensor
        
        
    def __getLabel__(self, imageName): 
        '''enter the name of the image and get the corresponding label'''
        #folderName= '../HAM10000_metadata.csv'
        dic = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
        with open(self.labels_dir, 'r' ) as csvfile:		
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')		
            next(reader)
            for row in reader:
                if imageName == row[1]:
                    #print('test: dic[row[2]]: ', dic[row[2]])
                    return dic[row[2]]
    
    # Aenderung    
    def __getitem__(self, idx):
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
        
            
        sample = {"image": image, "mask":mask}
        return sample



# Aenderung 1
''' ToTensor needs a PIL or np-arr. '''
''' transforms image to tensor,'''
''' applies colorJitter to image and normalizes image   '''
''' returns tensor with rgb layer and a forth layer which is the segmentation image '''
transform_data_seg_1 = transforms.Compose([
    transforms.ColorJitter(0.4,0.4,0.4),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[194.6979, 139.2626, 145.4852], std=[22.8551, 30.9032, 33.9032])
    transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
    ])
    
# Aenderung 3
''' ToTensor needs a PIL or np-arr. '''
''' transforms segmentation image to tensor,'''
transform_toTensor = transforms.ToTensor()
    
# Aenderung 4
transform_data_seg_2 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
    transforms.RandomPerspective(distortion_scale=0.2),
    ])
   
# Aenderung 5
transform_data_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
    ])

''' transforms input image '''   
transform_data_seg_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[194.6979, 139.2626, 145.4852], std=[22.8551, 30.9032, 33.9032])
    transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
    ])
