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

import argparse
import pickle

from customDataset import CustomImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train_test_split(
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

    with open( '../data/HAM10000_metadata.csv', 'r' ) as csvfile:		
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
        

def train_network(training_names, validation_names, test_names):

    # transformer (size, normalize make to Tensor)
    ''' ToTensor needs a PIL or np-arr. '''
    transform_data = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ColorJitter(0.4,0.4,0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
        ])   # mean and stddev have been previously calculated
        
    transform_val = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1212, 0.1330])
        ]) 

        
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
        
    #overfitDataset = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images/', transform = transform_val, list_im = overfit_names)




    resnet34 = models.resnet34(pretrained = True)
    num_ftrs = resnet34.fc.in_features
    resnet34.fc = torch.nn.Linear(num_ftrs, 7)

    resnet34.train()
    resnet34.to(device)

    #print(resnet34(imTensor.unsqueeze(0)))

    # ------------------------------------------------ train the model ------------------------------------------ #

    train_dl = DataLoader(myDataset, batch_size = 32, shuffle = True)
    #train_dl = DataLoader(overfitDataset, batch_size = 10, shuffle = True)

    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet34.parameters(), lr=0.00003) # stochastic gradient descent

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2074)


    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)

    #train the model


    testData = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images', transform = transform_val, list_im = test_names)
    test_dl = DataLoader(testData, batch_size = 100, shuffle = False)
    valData = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images', transform = transform_val, list_im = validation_names)
    val_dl = DataLoader(valData, batch_size = 100, shuffle = False) 
    #test_dl = DataLoader(overfitDataset, batch_size = 10, shuffle = False)
    print('len(testData: ', len(testData))

    epochs = 100
    best_bacc = 0
    early_stopping = 0
    for epoch in range(epochs):
       
        print("Validate the network before epoch {}".format(epoch))
        resnet34.eval()
        mat = np.zeros((7,7))
        for _, batch in tqdm(enumerate(val_dl)):
            output = resnet34(batch['image'].to(device)).detach().cpu().numpy()
            for j in range(output.shape[0]):
                mat[np.argmax(output[j]), batch['label'][j]] = mat[np.argmax(output[j]), batch['label'][j]] + 1

        print(mat)
        print(np.trace(mat) / np.sum(mat))
        bacc = 0
        mean_recall = 0
        for i in range(7):
            bacc = bacc + mat[i,i] / (7 * np.sum(mat[:,i]))
            mean_recall = mean_recall + mat[i,i] / (7 * np.sum(mat[i]) + 0.1)
        print(bacc)
        print(mean_recall)
        if bacc > best_bacc:    
            torch.save(resnet34.state_dict(), 'model_best.pt')
            best_bacc = bacc
            early_stopping = 0
        else:
            early_stopping = early_stopping + 1
            if early_stopping >= 10:
                print("stopped due to not improving during the last ten epochs")
                break
        print("Best accuracy at the moment: {}, ({} epochs until early stopping)".format(best_bacc, 10 - early_stopping))

        resnet34.train()
        epoch_loss = 0
        for i, batch in tqdm(enumerate(train_dl)):
            #print(i)
            #print(batch['label'])
            
            # clear the gradients
            optimizer.zero_grad()
            
            # compute the model outputs
            output_model = resnet34(batch['image'].to(device))
            
            # calc loss
            loss = criterion(output_model, batch['label'].to(device))

            # calculate derivative for each parameter
            loss.backward()
            
            # update model weights
            optimizer.step()
            scheduler.step()
       
        
    resnet34.eval()
    mat = np.zeros((7,7))
    for _, batch in tqdm(enumerate(test_dl)):
        output = resnet34(batch['image'].to(device)).detach().cpu().numpy()
        for j in range(output.shape[0]):
            mat[np.argmax(output[j]), batch['label'][j]] = mat[np.argmax(output[j]), batch['label'][j]] + 1

    print(mat)
    print(np.trace(mat) / np.sum(mat))
    bacc = 0
    mean_recall = 0
    for i in range(7):
        bacc = bacc + mat[i,i] / (7 * np.sum(mat[:,i]))
        mean_recall = mean_recall + mat[i,i] / (7 * np.sum(mat[i]) + 0.1)
    print(bacc)
    print(mean_recall)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Programm to train a neural'
            'on a the HAM10000 dataset')
    parser.add_argument('test_train_split', help = 'The .pkl file that contains'
            'the train and test split')
    
    args = parser.parse_args()
    try:
        with open(args.test_train_split, 'rb') as f:
            lists = pickle.load(f)
        training_names = lists['training_names']
        validation_names = lists['validation_names']
        test_names = lists['test_names']
    except:
        training_names, validation_names, test_names = train_test_split()
        lists = {}
        lists['training_names'] = training_names
        lists['validation_names'] = validation_names 
        lists['test_names'] = test_names 
        with open(args.test_train_split, 'wb') as f:
            pickle.dump(lists, f, pickle.HIGHEST_PROTOCOL)

    train_network(training_names, validation_names, test_names)
