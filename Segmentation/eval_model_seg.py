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
from train_model_seg import JaccardAccuracy, IaU, IoU, getModelAccuracy, pixel_accuracy
import segmentation_models_pytorch as smp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def ratio_backgroundVSlesion(batch_mask):
    ''' determines how many px are lesion and how many are background
    returns (count_lesion, count_background)
    count_lesion: No. of pixels that belong to lesion
    count_background: No. of pixels that belong to background
    '''
    count_background = 0
    count_lesion = 0
    n_masks = batch_mask.shape[0]
    for i in range(n_masks):
        count_lesion += (batch_mask[i] != 7).sum()
        count_background += (batch_mask[i] == 7).sum()
    return (count_lesion, count_background)
    
def acc_of_nobackground(batch_mask, batch_prediction, percentage=0.9):
    '''
    returns the number of predictions where at least the entered percentage of pixels
    are correctly determined under the condition that the pixels do not belong to background
    '''
    n_masks = batch_mask.shape[0]
    
    n_elements = (batch_mask != 7).sum(dim = [1,2,3])
    #print('n_elements: ', n_elements.shape)
    # Ab hier weiter machen.
    count = 0
    batch_prediction = torch.argmax(batch_prediction, dim = 1, keepdims = True)
    count_maskRatio = 0
    batch_mask[batch_mask==7] = 8 # change background so it is diff. to background of prediction
    
    
    #print(batch_mask[0])
    print(n_elements)
    print('percentage: ', percentage)
    print('n_elements[0]: ', n_elements[0])
    print('batch_mask[0] == batch_prediction[0]).sum(): ', (batch_mask[0] == batch_prediction[0]).sum().detach().numpy())
    print('percentage * n_elements[0]: ', percentage * n_elements[0])
    
    
    for i in range(n_masks):
        #print((batch_mask[i] == batch_prediction[i]).sum())
        if (batch_mask[i] == batch_prediction[i]).sum() >= percentage * n_elements[i]:
            count += 1
    
    print('count: ', count)
    print('')
    return (count)

def eval_model(resnet34, test_names, path_to_csv, path_to_images, path_to_masks):
 
    '''transform_val = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
        ]) 
    '''
    
    #testData = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images', transform = transform_val, list_im = test_names)    
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
    
    #mat = np.zeros((7,7))
    '''
    for batch in tqdm(iter(test_dl)):
        output = resnet34(batch['image'].to(device)).detach().cpu().numpy()
        for j in range(output.shape[0]):
            mat[np.argmax(output[j]), batch['label'][j]] = mat[np.argmax(output[j]), batch['label'][j]] + 1
    '''
    
    # accuracy
    ###print("Testset accuracy: ", getModelAccuracy(resnet34, test_dl))
    
    IaU_arr_val = np.zeros(14)
    count_pixAcc = [0,0,0]
    avg_px_lesion = 0
    count_maskratio = 0
    ratio_bgVSls = [0,0]
    count_nobackground90 = 0
    count_nobackground50 = 0
    
    for batch in tqdm(iter(test_dl)):
        output_val = resnet34(batch['image'].to(device))
        IaU_arr_val += IaU(batch['mask'], output_val)
        px_acc_tupel = pixel_accuracy(batch['mask'], output_val, 0.9)
        count_pixAcc[0] += px_acc_tupel[0]
        count_pixAcc[1] += px_acc_tupel[1]
        count_pixAcc[2] += px_acc_tupel[2]
        #print(count_pixAcc)
        ratio_mask_count = ratio_backgroundVSlesion(batch['mask'])
        ratio_bgVSls[0] += ratio_mask_count[0]
        ratio_bgVSls[1] += ratio_mask_count[1]
        
        count_nobackground90 += acc_of_nobackground(batch['mask'], output_val, 0.9)
        count_nobackground50 += acc_of_nobackground(batch['mask'], output_val, 0.5)
        '''if torch.equal(a,b):
            print('a==b')
        else: 
            print('a!=b')'''
        
    Jacc_val = JaccardAccuracy(IaU_arr_val)
    IoU_val = IoU(IaU_arr_val)
    
    print("Testset Jaccard score: ", Jacc_val)
    print("Testset IoU score: ", IoU_val)
    print("Pixel accuracy (0.9) (No_90): ", count_pixAcc[0]/count_pixAcc[1])
    print("Ratio lesion vs. background [ratio_lesion_background]: ", ratio_bgVSls[0] / ratio_bgVSls[1])
    print("No. of masks that meet conditon (No_lesion_less10): lesion <= 0.1 of whole image :", count_pixAcc[2] / count_pixAcc[1])
    print("No. of predictions where at least 0.9 of the lesion (w/o background) is predicted correctly: ", count_nobackground90/count_pixAcc[1])
    print("No. of predictions where at least 0.5 of the lesion (w/o background) is predicted correctly: ", count_nobackground50/count_pixAcc[1])





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Programm to train a neural'
            'on a the HAM10000 dataset')
    parser.add_argument('model_name', help = 'The model to be tested.')
    parser.add_argument('test_train_split', help = 'The .pkl file that contains'
            'the train and test split')
    parser.add_argument('--set', help = 'Which set to test. Must be in "val", "train" and "test"')
    
    # input path to csv file containing image names including filename (!!!)
    parser.add_argument('path_to_csv', help = 'Input path to csv file containing image names.')
    
    # input path to images
    parser.add_argument('path_to_images', help = 'Input path to images.')
    
    # input path to masks
    parser.add_argument('path_to_masks', help = 'Input path to masks.')
    
    args = parser.parse_args()
    try:
        with open(args.test_train_split, 'rb') as f:
            lists = pickle.load(f)
    except:
        print("Could not open the pikel containing the training_test_split")
        exit()
    
    if args.set == 'val':
        test_names = lists['validation_names']
    elif args.set == 'train':
        test_names = lists['training_names']
    else:
        test_names = lists['test_names']

    try:
        #resnet34 = models.resnet34(pretrained = False)
        #num_ftrs = resnet34.fc.in_features
        #resnet34.fc = torch.nn.Linear(num_ftrs, 7)
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
        
    eval_model(resnet34, test_names, args.path_to_csv, args.path_to_images, args.path_to_masks)
