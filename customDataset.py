from torch.utils.data import Dataset
from torchvision import transforms

import PIL
import csv

class CustomImageDataset(Dataset):
    

    def __init__(self, metaDataFile, img_dir, transform = None, list_im = None):
        ''' img_dir = path to images, metaDataFile: path to labels'''
        self.img_dir = img_dir
        self.labels_dir = metaDataFile
        self.transform = transform
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
                    
    def __getitem__(self, idx):
        im = PIL.Image.open('../data/images/{}.jpg'.format(self.img_list[idx]))
        #image = read_image(self.img_dir) # Bild soll mit idx aufgerufen werden (in csv Liste)  -> self parameter ist die Liste. statt Panda nutze PIL
        label = self.__getLabel__(self.img_list[idx])
        if self.transform:
            image = self.transform(im)
            
        sample = {"image": image, "label":label}
        return sample


def transform_pg(level = 0):
    """
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
        transforms.ColorJitter(0.4,0.4,0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
        ])   # mean and stddev have been previously calculated
    return transform_data

