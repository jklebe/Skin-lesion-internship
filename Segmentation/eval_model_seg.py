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

def ratio_backgroundVSlesion(batch_mask):
    ''' determines how many px are lesion and how many are background
    returns (count_lesion, count_background)
    count_lesion: No. of pixels that belong to lesion
    count_background: No. of pixels that belong to background
    '''
    
    count_lesion = (batch_mask != 7).sum()
    count_background = (batch_mask == 7).sum()
    
    return (count_lesion, count_background)
    
def acc_of_nobackground(batch_mask_tensor, batch_prediction_tensor, percentage=0.9):
    '''
    returns the number of predictions where at least the entered percentage of pixels
    are correctly determined under the condition that the pixels do not belong to background
    '''

    batch_mask = batch_mask_tensor.cpu().detach().numpy()
    batch_prediction = batch_prediction_tensor.cpu().detach().numpy()
    batch_mask = copy.copy(batch_mask)
    batch_prediction = copy.copy(batch_prediction)

    #defining some variables
    n_masks = batch_mask.shape[0]
    n_elements = np.sum(np.sum(np.sum(batch_mask != 7, axis = 1), axis = 1), axis = 1)
    count = 0
    count_maskRatio = 0
    
    #prepare pred and mask
    batch_prediction = np.expand_dims(np.argmax(batch_prediction, axis = 1), axis = 1)
    batch_mask[batch_mask==7] = -1 # change background so it is diff. to background of prediction
     
    for i in range(n_masks):
        if (batch_mask[i] == batch_prediction[i]).sum() >= percentage * n_elements[i]:
            count += 1
    
    return count
    
def IaU_maliciousClass(batch_mask, batch_prediction):
    ''' 
    determines intersection and union of mask and prediction 
    returns np.array with 15 entries: first 7 entries are intersection for classes 0 to 6
    entries 7 to 13: values of union for classes 0 to 6 
    
    modification: bcc and mel have same class and returns arr with IoU.
    '''
    arr = np.zeros(14)
    batch_prediction = torch.argmax(batch_prediction, dim = 1, keepdims = True)
    
    batch_prediction_new = copy.deepcopy(batch_prediction)
    batch_mask_new = copy.deepcopy(batch_mask)
    
    batch_prediction_new[batch_prediction_new==4] = 1
    batch_mask_new[batch_mask_new==4] = 1
    
    
    for i in range(7):
    
        prediction_akiec = torch.zeros_like(batch_prediction_new)
        prediction_akiec[batch_prediction_new == i] = 1
        
        mask_akiec = torch.zeros_like(batch_mask_new)
        mask_akiec[batch_mask_new == i] = 1
        

        intersection_akiec = torch.sum(prediction_akiec * mask_akiec)
        union_akiec = torch.sum(prediction_akiec) + torch.sum(mask_akiec) - intersection_akiec
        
        arr[i] = intersection_akiec
        arr[i + 7] = union_akiec
        
    return arr
    
    
    

def IaU_noClass(batch_mask, batch_prediction):
    ''' 
    determines intersection and union of mask and prediction 
    returns np.array with 15 entries: first 7 entries are intersection for classes 0 to 6
    entries 7 to 13: values of union for classes 0 to 6 
    
    modification: no classes only lesion and background.
    '''
    arr = np.zeros(2)
    batch_prediction = torch.argmax(batch_prediction, dim = 1, keepdims = True)
    
    batch_prediction_new = copy.deepcopy(batch_prediction)
    batch_mask_new = copy.deepcopy(batch_mask)
    
    batch_prediction_new[batch_prediction_new != 7] = 1
    batch_prediction_new[batch_prediction_new == 7] = 0
    
    mask_akiec = torch.zeros_like(batch_mask_new)
    mask_akiec[batch_mask_new != 7] = 1
    

    intersection_akiec = torch.sum(batch_prediction_new * mask_akiec)
    union_akiec = torch.sum(batch_prediction_new) + torch.sum(mask_akiec) - intersection_akiec
        
    arr[0] = intersection_akiec
    arr[1] = union_akiec
        
    return arr


def IoU_noClass(arr):
    ''' takes the array of func IaU and returns arr of IoU for each class '''
    
    return arr[0]/arr[1]



    
def make_IoUArr_mal(IaU_arr):
    ''' takes the array of func IaU and returns arr of IoU for each class '''#
    arr = np.zeros(7)
    
    for i in range(7):
        if i != 4: 
            arr[i] = IaU_arr[i]/IaU_arr[i+7]
        else:
            arr[i] = 0
        
    return arr
    

def eval_model(resnet34, test_names, path_to_csv, path_to_images, path_to_masks, debug = True):
    
    if debug:
        print('start eval model')
    
    # Creatung Dataset and Data Loader
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
    


    #start the tests
    IaU_arr_val = np.zeros(14)
    IaU_arr_malClass = np.zeros(14)
    IaU_arr_noClass = np.zeros(2)
    
    count_pixAcc = [0,0,0]
    avg_px_lesion = 0
    count_maskratio = 0
    ratio_bgVSls = [0,0]
    count_nobackground90 = 0
    count_nobackground50 = 0
    
    for batch in tqdm(iter(test_dl)):
        output_val = resnet34(batch['image'].to(device))
        IaU_arr_val += IaU(batch['mask'].clone(), output_val.clone())
        IaU_arr_malClass += IaU_maliciousClass(batch['mask'].clone(), output_val.clone())
        IaU_arr_noClass += IaU_noClass(batch['mask'].clone(), output_val.clone())
        
        px_acc_tupel = pixel_accuracy(batch['mask'].clone(), output_val.clone(), 0.9)
        count_pixAcc[0] += px_acc_tupel[0]
        count_pixAcc[1] += px_acc_tupel[1]
        count_pixAcc[2] += px_acc_tupel[2]
        #print(count_pixAcc)
        ratio_mask_count = ratio_backgroundVSlesion(batch['mask'].clone())
        ratio_bgVSls[0] += ratio_mask_count[0]
        ratio_bgVSls[1] += ratio_mask_count[1]
        
        count_nobackground90 += acc_of_nobackground(batch['mask'].clone(), output_val.clone(), 0.9)
        count_nobackground50 += acc_of_nobackground(batch['mask'].clone(), output_val.clone(), 0.5)
        
    Jacc_val = JaccardAccuracy(IaU_arr_val)
    IoU_val = IoU(IaU_arr_val)
    
    #print("Testset accuracy: ", getModelAccuracy(resnet34, test_dl))
    print("Testset Jaccard score: ", Jacc_val)
    print("IoU array: ", make_IoUArr(IaU_arr_val))
    print("IoU array with one mal group: ", make_IoUArr_mal(IaU_arr_malClass))
    print("IoU array with no class: ", IoU_noClass(IaU_arr_noClass))
    print("Testset IoU score: ", IoU_val)
    print("Pixel accuracy (0.9) (No_90): ", count_pixAcc[0]/count_pixAcc[1])
    print("Ratio lesion vs. background [ratio_lesion_background]: ", ratio_bgVSls[0] / ratio_bgVSls[1])
    print("No. of masks that meet conditon (No_lesion_less10): lesion <= 0.1 of whole image :", count_pixAcc[2] / count_pixAcc[1])
    print("No. of predictions where at least 0.9 of the lesion (w/o background) is predicted correctly: ", count_nobackground90/count_pixAcc[1])
    print("No. of predictions where at least 0.5 of the lesion (w/o background) is predicted correctly: ", count_nobackground50/count_pixAcc[1])





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
        print("could not load the model")
        exit()
    
    print("start eval model")
    eval_model(resnet34, test_names, args.path_to_csv, args.path_to_images, args.path_to_masks)
