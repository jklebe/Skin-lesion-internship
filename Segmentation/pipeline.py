import argparse
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

import customDatasetSeg
import train_model_seg
from customDatasetSeg import CustomImageDataset
from train_model_seg import JaccardAccuracy, IaU, IoU, getModelAccuracy, pixel_accuracy, make_IoUArr
import segmentation_models_pytorch as smp

import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




def eval_model(resnet34, test_names, path_to_csv, path_to_images, path_to_masks, debug = True):
    
    if debug:
        print('start eval model')
    
    # Creating Dataset and Data Loader
    testData = CustomImageDataset(
        path_to_csv,
        path_to_images,
        path_to_masks,
        transform_data_seg_1 = customDatasetSeg.transform_data_seg_test,
        transform_toTensor = customDatasetSeg.transform_toTensor,
        transform_data_seg_2 = customDatasetSeg.transform_data_val,
        list_im = test_names
        )
    test_dl = DataLoader(testData, batch_size = 32, shuffle = False)

    resnet34.eval()
    resnet34.to(device)
    
    noClasses = 8
    No_images = len(testData)
    avg = np.zeros((No_images, noClasses))
    all_count_labelMatches = np.zeros((No_images, noClasses))
    all_sum_labelMatches = np.zeros((No_images, noClasses))
    list_names = []

    count_image = 0
    for batch in tqdm(iter(test_dl)):
        output_val = resnet34(batch['image'].to(device))
        #print('output_val.shape: ', output_val.shape)

        output_argmax = np.expand_dims(np.argmax(output_val.cpu().detach().numpy(), axis = 1), axis = 1)
        
        for i_image in range(batch['image'].shape[0]):    # iterate through images in batch
            count_labelMatches = np.zeros(noClasses)
            sum_labelMatches = np.zeros(noClasses)
           
            for nr_cls in range(7):
                all_count_labelMatches[count_image][nr_cls] = np.sum(np.where(output_argmax[i_image] == nr_cls, 1, 0)) + 1
                all_sum_labelMatches[count_image][nr_cls] = np.sum(np.where(output_argmax[i_image] == nr_cls, output_val[i_image][nr_cls].cpu().detach().numpy(), 0))
            
            count_image += 1
            list_names.append(batch['name'][i_image])

    avg = all_sum_labelMatches.astype(np.float32)/all_count_labelMatches.astype(np.float32)
            

    # determine max for each class
    print('avg.shape: ', avg.shape)  
    avg_argmax = np.argmax(avg, axis = 0)
    print(avg_argmax)
    
    for i in range(count_image):
        print('Label: ', i, '\t Image: ', list_names[avg_argmax[i]])

        
        
     

        
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Programm to train a neural'
            'on a the HAM10000 dataset')
    
    #Add parser arguments
    parser.add_argument('model_name', help = 'The model to be tested.')
    parser.add_argument('test_train_split', help = 'The .pkl file that contains'
            'the train and test split')
    parser.add_argument('path_to_csv', help = 'Input path to csv file'
            'containing image names including filename (!!!).')
    parser.add_argument('path_to_images', help = 'Input path to images.')
    parser.add_argument('path_to_masks', help = 'Input path to masks.')
    parser.add_argument('--set', help = 'Which set to test. Must be in "val", '
            '"train" and "test"')
    
    
    args = parser.parse_args()
    try:
        with open(args.test_train_split, 'rb') as f:
            lists = pickle.load(f)
    except:
        print("Could not open the pikel containing the training_test_split")
        exit()
    
    if args.set == 'val':
        test_names = lists['validation_names']
        print('Testing on validation set')
    elif args.set == 'train':
        test_names = lists['training_names']
        print('Testing on training set')
    else:
        test_names = lists['test_names']
        print('Testing on test set')


    try:
        resnet34 = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=8,                      # model output channels (number of classes in your dataset)
        )

        resnet34.load_state_dict(torch.load(args.model_name, map_location=device))
    except:
        print("could not load the model: ", args.model_name)
        exit()
    
    print("start eval model")
    eval_model(resnet34, test_names, args.path_to_csv, args.path_to_images, args.path_to_masks)
