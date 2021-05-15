import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import torchvision.models as models
import PIL
from torchvision import transforms
import torch.nn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

import argparse
import pickle

import customDatasetSeg
from customDatasetSeg import CustomImageDataset

from lossmulti import LossMulti

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
PATH = '../../data/'

def train_test_split(
        path_to_csv,
        test_set_percentage = 0.2, 
        validation_set_percentage = 0.1,
        show_plot = False,
        debug = False):

    names = []
    labels = []
    train_set_percentage = 1 - test_set_percentage - validation_set_percentage

    # maps labels in numbers
    dic = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}

    if debug:
        test = 'nv'
        print(dic[test])

    with open(path_to_csv, 'r' ) as csvfile:		
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')		
        next(reader)
        for row in reader:
            names.append(row[1])
            labels.append(dic[row[2]])
    File_length = len(names)
    
    if debug:
        print(File_length)
        print(labels)

    if show_plot:
        # Show Histogram of Data
        plt.xticks([0, 1, 2, 3, 4, 5, 6], ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
        plt.xlabel('Labels')
        plt.ylabel('Count / Häufigkeit')
        plt.hist(labels, bins=[0-0.5,1-0.5,2-0.5,3-0.5,4-0.5,5-0.5,6-0.5,7-0.5]) # create histogram
        plt.show()

    # determine the frequency of the labels

    labels_count = {}

    # iterating over the labels for frequency
    for l in labels:
       # checking whether it is in the dict or not
       if l in labels_count:
          # incerementing the count by 1
          labels_count[l] += 1
       else:
          # setting the count to 1
          labels_count[l] = 1
          
    # printing the elements frequencies
    if debug:
        for key in range(7):
            print(f"{key}: {labels_count[key]}")


    # compute training and test set
    print('Anzahl aller labels: ', len(labels))

    # lists for sets accoring to labels
    names_akiec = []
    names_bcc = []
    names_bkl = []
    names_df = []
    names_mel = []
    names_nv = []
    names_vasc = []

    reversedict = {0: names_akiec, 1: names_bcc, 2: names_bkl, 3: names_df, 4: names_mel, 5: names_nv, 6: names_vasc}

    # fill lists with image names
    for i in range (len(names)):
        reversedict[labels[i]].append(names[i])
        
    # test length
    print(len(names_akiec))
    print(len(names_bcc))
    print(len(names_bkl))
    print(len(names_df))
    print(len(names_mel))
    print(len(names_nv))
    print(len(names_vasc))

    # shuffle lists
    random.shuffle(names_akiec)
    random.shuffle(names_bcc)
    random.shuffle(names_bkl)
    random.shuffle(names_df)
    random.shuffle(names_mel)
    random.shuffle(names_nv)
    random.shuffle(names_vasc)

    # create training set
    training_names_akiec = names_akiec[:int(len(names_akiec)*(train_set_percentage))]
    training_names_bcc = names_bcc[:int(len(names_bcc)*(train_set_percentage))]
    training_names_bkl = names_bkl[:int(len(names_bkl)*(train_set_percentage))]
    training_names_df = names_df[:int(len(names_df)*(train_set_percentage))]
    training_names_mel = names_mel[:int(len(names_mel)*(train_set_percentage))]
    training_names_nv = names_nv[:int(len(names_nv)*(train_set_percentage))]
    training_names_vasc = names_vasc[:int(len(names_vasc)*(train_set_percentage))]

    validation_names_akiec = names_akiec[int(len(names_akiec)*(train_set_percentage)):int(len(names_akiec)*(train_set_percentage + validation_set_percentage))]
    validation_names_bcc = names_bcc[int(len(names_bcc)*(train_set_percentage)):int(len(names_bcc)*(train_set_percentage + validation_set_percentage))]
    validation_names_bkl = names_bkl[int(len(names_bkl)*(train_set_percentage)):int(len(names_bkl)*(train_set_percentage + validation_set_percentage))]
    validation_names_df = names_df[int(len(names_df)*(train_set_percentage)):int(len(names_df)*(train_set_percentage + validation_set_percentage))]
    validation_names_mel = names_mel[int(len(names_mel)*(train_set_percentage)):int(len(names_mel)*(train_set_percentage + validation_set_percentage))]
    validation_names_nv = names_nv[int(len(names_nv)*(train_set_percentage)):int(len(names_nv)*(train_set_percentage + validation_set_percentage))]
    validation_names_vasc = names_vasc[int(len(names_vasc)*(train_set_percentage)):int(len(names_vasc)*(train_set_percentage + validation_set_percentage))]



    # create testing sets
    test_names_akiec = names_akiec[int(len(names_akiec)*(train_set_percentage + validation_set_percentage)):]
    test_names_bcc = names_bcc[int(len(names_bcc)*(train_set_percentage + validation_set_percentage)):]
    test_names_bkl = names_bkl[int(len(names_bkl)*(train_set_percentage + validation_set_percentage)):]
    test_names_df = names_df[int(len(names_df)*(train_set_percentage + validation_set_percentage)):]
    test_names_mel = names_mel[int(len(names_mel)*(train_set_percentage + validation_set_percentage)):]
    test_names_nv = names_nv[int(len(names_nv)*(train_set_percentage + validation_set_percentage)):]
    test_names_vasc = names_vasc[int(len(names_vasc)*(train_set_percentage + validation_set_percentage)):]


    if debug:
        #check
        print('')
        print(len(training_names_akiec) + len(test_names_akiec))
        print(len(training_names_bcc) + len(test_names_bcc))
        print(len(training_names_bkl) + len(test_names_bkl))
        print(len(training_names_df) + len(test_names_df))
        print(len(training_names_mel) + len(test_names_mel))
        print(len(training_names_nv) + len(test_names_nv))
        print(len(training_names_vasc) + len(test_names_vasc))

    training_names = 20 * training_names_akiec + 13 * training_names_bcc + 6 * training_names_bkl + 58 * training_names_df + 6 * training_names_mel + training_names_nv + 47 * training_names_vasc
    #training_names = training_names_akiec + training_names_bcc + training_names_bkl + training_names_df + training_names_mel + training_names_nv + training_names_vasc
    if debug:
        print('len(training_names) :', len(training_names))

    validation_names = validation_names_akiec+validation_names_bcc+validation_names_bkl+validation_names_df+validation_names_mel+validation_names_nv+validation_names_vasc

    test_names = test_names_akiec+test_names_bcc+test_names_bkl+test_names_df+test_names_mel+test_names_nv+test_names_vasc
    if debug:
        print(len(test_names))
    
    return training_names, validation_names, test_names


# ---------------------- Modell ---------------------------------------------------- #

# determine avg and stddev of images
#for name in training_names_akiec + training_names_bcc + training_names_bkl + training_names_df + training_names_mel + training_names_nv +training_names_vasc:
#    imageio.imread('../images/{}.jpg' .format((name))

# Training Set


#overfit_names = [training_names_akiec[0], 
#                 training_names_bcc[0],
#                 training_names_bkl[0],
#                 training_names_df[0],
#                 training_names_mel[0],
#                 training_names_nv[0],
#                 training_names_vasc[0]]
 

def getModelAccuracy(model, set_dl):
    ''' determines how many pixels of the output and the mask are same '''
    ''' returns average value  '''
    try: 
        model.eval()
    except:
        pass
    count_batch = 0
    acc_seg = 0
    for batch_ex in tqdm(iter(set_dl)):
        prediction = model(batch_ex['image'].to(device))
        prediction = torch.argmax(prediction, dim = 1, keepdims = True)
        prediction = prediction.cpu().detach().numpy()
        mask = batch_ex['mask'].cpu().detach().numpy()
        acc_seg += np.sum(prediction == mask) / (np.sum(np.ones_like(prediction)))
        count_batch += 1
        
    acc_seg = acc_seg/ count_batch
    
    return acc_seg
    
def JaccardAccuracy(IaU_arr):
    ''' determines the Jaccard coeff of all predictions at once
    using sum of the outputs of the IaU func ( = IaU_arr) '''
    jacc = 0
    #print(IaU_arr)
    for i in range(7):
        IoU =  IaU_arr[i] / IaU_arr[i+7]
        if IoU <= 0.65:
            IoU = 0
        jacc += IoU
    jacc = jacc / 7.0
    return jacc
    
def IoU(IaU_arr):
    jacc = 0
    for i in range(7):
        jacc +=  IaU_arr[i] / IaU_arr[i+7]
    jacc = jacc / 7.0
    return jacc
   
    
def IaU(mask, prediction):
    ''' 
    determines intersection and union of mask and prediction 
    returns np.array with 15 entries: first 7 entries are intersection for classes 0 to 6
    entries 7 to 13: values of union for classes 0 to 6 
    '''
    arr = np.zeros(14)
    prediction = torch.argmax(prediction, dim = 1, keepdims = True)
    
    
    
    for i in range(7):
    
        prediction_akiec = torch.zeros_like(prediction)
        prediction_akiec[prediction == i] = 1
        
        mask_akiec = torch.zeros_like(mask)
        mask_akiec[mask == i] = 1
        

        intersection_akiec = torch.sum(prediction_akiec * mask_akiec)
        union_akiec = torch.sum(prediction_akiec) + torch.sum(mask_akiec) - intersection_akiec
        
        arr[i] = intersection_akiec
        arr[i + 7] = union_akiec
        
    return arr 
    
def make_IoUArr(IaU_arr):
    ''' takes the array of func IaU and returns arr of IoU for each class '''#
    arr = np.zeros(7)
    
    for i in range(7):
        arr[i] = IaU_arr[i]/IaU_arr[i+7]
        
    return arr
    
def pixel_accuracy(batch_mask, batch_prediction, percent):
    '''
    Compares each pixel. Condition: if percent % of the pixels between mask and
    prediction are same, adds to count. Function returns 
    1. count: number of predictions which meet condition above.
    2. n_mask: number of masks in batch batch_mask.
    3. count_maskRation: number of masks where lesion take < (1-percent) of image
    '''
    n_masks = batch_mask.shape[0]
    n_elements = torch.numel(batch_mask[0])
    count = 0
    batch_prediction = torch.argmax(batch_prediction, dim = 1, keepdims = True)
    count_maskRatio = 0
    
    for i in range(n_masks):
        if (batch_mask[i] == batch_prediction[i]).sum() >= percent * n_elements:
            count += 1
        if (batch_mask[i] != 7).sum() < (1 - percent) * n_elements:
            count_maskRatio += 1
    
    return (count, n_masks, count_maskRatio)

def train_network(training_names, validation_names, test_names, path_to_csv, path_to_images, path_to_masks):
    '''    
    im = PIL.Image.open("../data/images/ISIC_0024306.jpg")
    imTensor = transform_data(im)
    print(imTensor.shape)
    print(np.std(imTensor.numpy(), axis=(1,2)))
    print(imTensor.numpy().shape)

    myDataset = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images', transform = transform_data, list_im = training_names)
    print('GETITEM: ')
    example = myDataset.__getitem__(1)
    print(example['image'].shape) #image richtige und label
    print('len(myDataset): ', len(myDataset))
    '''
        
    #overfitDataset = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images/', transform = transform_val, list_im = overfit_names)


    # Aenderungen
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=8,                      # model output channels (number of classes in your dataset)
    )
    #-------------------------------------------------------#
    #-------------------------------------------------------#
    #-------------------------------------------------------#
    
    model.train()
    model.to(device)

    trainDataset = CustomImageDataset(
            path_to_csv,
            path_to_images,
            path_to_masks,
            transform_data_seg_1 = customDatasetSeg.transform_data_seg_1,
            transform_toTensor = customDatasetSeg.transform_toTensor,
            transform_data_seg_2 = customDatasetSeg.transform_data_seg_2,
            list_im = training_names
            )
    
    '''
    ex = trainDataset.__getitem__(0)
    #transforms.ToPILImage()(ex['image']).show()
    #transforms.ToPILImage()(ex['mask']).show()
    output_classes = torch.argmax(model(ex['image'].unsqueeze(0)), dim = 1)
    output_classes = output_classes.squeeze().detach().numpy()
    plt.figure()
    plt.imshow(output_classes)
    plt.show()
    plt.close()
    '''

    # ------------------------------------------------ train the model ------------------------------------------ #

    train_dl = DataLoader(trainDataset, batch_size = 32, shuffle = True)
    #train_dl = DataLoader(overfitDataset, batch_size = 10, shuffle = True)




    # define the optimization
    # criterion = smp.utils.losses.JaccardLoss()
    ###criterion = nn.CrossEntropyLoss()
    criterion = LossMulti(jaccard_weight=0.5, class_weights=None, num_classes=7)
    optimizer = optim.AdamW(model.parameters(), lr=0.00003) # stochastic gradient descent

    epoch_len = len(iter(train_dl))
    print("epoch_len: ", epoch_len)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_len) # 1037 = 1 epoch

    
    #testData = CustomImageDatasetSeg(PATH +'HAM10000_metadata.csv', PATH +'images', transform = transform_val, list_im = test_names)
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
    #valData = CustomImageDatasetSeg(PATH + 'HAM10000_metadata.csv', PATH + 'images', transform = transform_val, list_im = validation_names)
    valData = CustomImageDataset(
        path_to_csv,
        path_to_images,
        path_to_masks,
        transform_data_seg_1 = customDatasetSeg.transform_data_seg_test,
        transform_toTensor = customDatasetSeg.transform_toTensor,
        transform_data_seg_2 = customDatasetSeg.transform_data_val,
        list_im = validation_names
        )
    val_dl = DataLoader(valData, batch_size = 32, shuffle = False) 
    #test_dl = DataLoader(overfitDataset, batch_size = 10, shuffle = False)
    print('len(testData): ', len(testData))

    epochs = 100
    #best_bacc = 0
    early_stopping = 0
    best_Jacc = 0
    best_IoU = 0
    
    
    
    for epoch in range(epochs):
        
        IaU_arr = np.zeros(14)
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(iter(train_dl)):
        
            #print('batch[image].shape: ', batch['image'].shape)
            #print('batch[mask].shape: ', batch['mask'].shape)
            
            #print(i)
            #print(batch['label'])
            
            # clear the gradients
            optimizer.zero_grad()
            
            # compute the model outputs
            output_model = model(batch['image'].to(device))
            #print('output_model.shape: ', output_model.shape)
            
            # calc loss
            loss = criterion(output_model, batch['mask'].to(device).squeeze())

            epoch_loss = epoch_loss + loss
            # calculate derivative for each parameter
            loss.backward()
            
            # update model weights
            optimizer.step()
            scheduler.step()
            
            IaU_arr += IaU(batch['mask'], output_model)

        print(epoch_loss)
        
        print("Validation Epoch's acc: ", getModelAccuracy(model, val_dl))
        print("Training Epoch's validation Jaccard score: ", JaccardAccuracy(IaU_arr))
        
        # Validation set
        print("Validate the network after epoch {} on val. set ".format(epoch))
        model.eval()

        IaU_arr_val = np.zeros(14)
        for batch in tqdm(iter(val_dl)):
            output_val = model(batch['image'].to(device))
            IaU_arr_val += IaU(batch['mask'], output_val)
        Jacc_val = JaccardAccuracy(IaU_arr_val)
        IoU_val = IoU(IaU_arr_val)
        
        
        
        if IoU_val > best_IoU:    
            torch.save(model.state_dict(), 'model_best_seg.pt')
            best_IoU = IoU_val
            early_stopping = 0
        else:
            early_stopping = early_stopping + 1
            if early_stopping >= 10:
                print("stopped due to not improving during the last ten epochs")
                break
        print("Best IoU at the moment: {}, ({} epochs until early stopping)".format(best_IoU, 10 - early_stopping))       
        
    # test set
    print("Test the network after epoch {} on test set ".format(epoch))
    model.load_state_dict(torch.load('model_best_seg.pt', map_location=device))
    model.eval()

    IaU_arr_val = np.zeros(14)
    for batch in tqdm(iter(test_dl)):
        output_val = model(batch['image'].to(device))
        IaU_arr_val += IaU(batch['mask'], output_val)
    Jacc_val = JaccardAccuracy(IaU_arr_val)   
    print('Final Jaccard score: ', JaccardAccuracy(IaU_arr_val))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Programm to train a neural'
            'on a the HAM10000 dataset')
    parser.add_argument('test_train_split', help = 'The .pkl file that contains'
            'the train and test split')
    
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
        training_names = lists['training_names']
        validation_names = lists['validation_names']
        test_names = lists['test_names']
    except:
        training_names, validation_names, test_names = train_test_split(args.path_to_csv)
        lists = {}
        lists['training_names'] = training_names
        lists['validation_names'] = validation_names 
        lists['test_names'] = test_names 
        with open(args.test_train_split, 'wb') as f:
            pickle.dump(lists, f, pickle.HIGHEST_PROTOCOL)

    train_network(training_names, validation_names, test_names, args.path_to_csv, args.path_to_images, args.path_to_masks)


# documentation
#run:
#train_model_seg.py test_train_split.pkl path_to_csv path_to_images path_to_masks
# if no test_train_split available, then a new file with a new training, validation and test set is created
