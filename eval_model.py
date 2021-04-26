import argparse
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms


from customDataset import CustomImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def eval_model(resnet34, test_names):
 
    transform_val = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[194.6979, 139.2626, 145.4852], std=[22.8551, 30.9032, 33.9032])
        ]) 
    
    testData = CustomImageDataset('../data/HAM10000_metadata.csv', '../data/images', transform = transform_val, list_im = test_names)
    test_dl = DataLoader(testData, batch_size = 100, shuffle = False)

    resnet34.eval()
    resnet34.to(device)
    mat = np.zeros((7,7))
    for _, batch in tqdm(enumerate(test_dl)):
        output = resnet34(batch['image'].to(device)).detach().cpu().numpy()
        for j in range(output.shape[0]):
            mat[np.argmax(output[j]), batch['label'][j]] = mat[np.argmax(output[j]), batch['label'][j]] + 1

    print(mat.astype(int))
    print('Accuracy: {}'.format(np.trace(mat) / np.sum(mat)))
    bacc = 0
    mean_recall = 0
    for i in range(7):
        bacc = bacc + mat[i,i] / (7 * np.sum(mat[:,i]))
        mean_recall = mean_recall + mat[i,i] / (7 * np.sum(mat[i]) + 0.1)
    print('Balanced Accuracy: {}'.format(bacc))
    print('Mean Recall: {}'.format(mean_recall))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Programm to train a neural'
            'on a the HAM10000 dataset')
    parser.add_argument('model_name', help = 'The model to be tested')
    parser.add_argument('test_train_split', help = 'The .pkl file that contains'
            'the train and test split')
    parser.add_argument('--set', help = 'Which set to test. Must be in "val", "train" and "test"')
    
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
        resnet34 = models.resnet34(pretrained = False)
        num_ftrs = resnet34.fc.in_features
        resnet34.fc = torch.nn.Linear(num_ftrs, 7)

        resnet34.load_state_dict(torch.load(args.model_name, map_location=device))
    except:
        print("could not load the model")

    eval_model(resnet34, test_names)
