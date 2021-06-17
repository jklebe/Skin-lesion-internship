# Skin-lesion Internship 

In this directory the code and other files for the classification task can be found.
The code for the classification code presumes that the data (csv files with labels and directory with images) is one directory hierarchy higher than the code. 
For the segmentation task the paths to the csv file with labels, to the directory with images and to the directoy with masks can be entered to the terminal/shell/console.

| File          | Content       |
| ------------- |-------------|
| train_model.py      | trains the model |
| train_test_split.pkl      | contains training, validation and test sets     |
| model_best_second_try.pt | contains one of the pre-trained models; because of limited storage capactiy not all models are uploaded   |
| eval_model.py |   if started, evaluates the model on either the training, validation or test set  |
| customDataset.py |  contains custom Image Dataset   |

## New model on existing split sets
If new model should be trained on existing split set, run
```
python train_model.py train_test_split.pkl
```
output: model_best.pt


## Evaluation of model
If model should be evaluated on test set, run 
```
python eval_model.py model_best_second_try.pt train_test_split.pkl --set test
```
If model should be evaluated on validation set, run 
```
python eval_model.py model_best_second_try.pt train_test_split.pkl --set val
```
If model should be evaluated on training set, run 
```
python eval_model.py model_best_second_try.pt train_test_split.pkl --set train
```



## New creation of split dataset
If newly split training, validation and test sets should be created, run
```
python train_model.py newNameFileForSavingDatasets.pkl
```


