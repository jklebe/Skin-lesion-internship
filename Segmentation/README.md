# Skin-lesion Internship 

| File          | Content       |
| ------------- |-------------|
| train_model_seg.py      | trains the model |
| train_test_split.pkl      | contains training, validation and test sets     |
| customDatasetSeg.py |  contains custom Image Dataset   |
| postprocessing.py |  contains postprocessing functions   |
| pipeline.py |  contains code to determine images which are ranked highest for a respective class   |
| eval_model.py |   evaluates the model on either the training, validation or test set  |

## Train model
To train model, run
```
python train_model_seg.py fileNametoSaveSubsets.pkl pathToCSVFileIncludingFilename pathToImages pathToMask
```

example:
```
python train_model_seg.py test_train_split_testData.pkl ../../data/test_HAM10000_metadata_new.csv ../../data/test_images ../../data/test_segmentation
```

If 
```
--pg
```
is appended, then progressive growing is applied.

If 
```
-cw "pathToWeights"
```
(where pathToWeights = path to the weighst, that should be used for the encoder) is appended, customweights can be used to initialization.


## Evaluation of model
To evaluate model on test set, run
```
python eval_model.py model_best.pt train_test_split.pkl path_to_csv path_to_images path_to_masks --set "setname"
```
where "setname" can be "train", "val" or "test", and refers to the set upon which the code is executed.

If 
```
--linknet
```
is appended, instead of Unet linknet is used.

If 
```
--pp
```
is appended, postprocessing is applied. Details on the postprocessing procedure can be found in postprocessing.py.