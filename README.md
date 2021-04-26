# Skin-lesion Internship 

| File          | Content       |
| ------------- |-------------|
| erste_Aufgabe.py      | trains the model |
| train_test_split.pkl      | contains training, validation and test sets     |
| model_best_second_try.pt | contains pre-trained model      |
| eval_model.py |   if started, evaluates the model on either the training, validation or test set  |
| customDataset.py |  contains custom Image Dataset   |


If new model should be trained on existing split set, run
```
python erste_Aufgabe.py train_test_split.pkl
```
output: model_best.pt



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




If newly split training, validation and test sets should be created, run
```
python erste_Aufgabe.py newNameFileForSavingDatasets.pkl
```


