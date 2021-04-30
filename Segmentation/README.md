# Skin-lesion Internship 

| File          | Content       |
| ------------- |-------------|
| train_model_seg.py      | trains the model |
| train_test_split.pkl      | contains training, validation and test sets     |
| customDatasetSeg.py |  contains custom Image Dataset   |

## To train model, run
```
python train_model_seg.py fileNametoSaveSubsets.pkl pathToCSVFileIncludingFilename pathToImages pathToMask
```

example:
```
python train_model_seg.py test_train_split_testData.pkl ../../data/test_HAM10000_metadata_new.csv ../../data/test_images ../../data/test_segmentation
```