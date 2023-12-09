# UBC-Ovarian-Cancer-subtypE-Classification
## Topic overview
https://www.kaggle.com/competitions/UBC-OCEAN/overview

## Dataset
- The thumbnails images that are separated into 5 different classes can be found in the **train_thumbnails** folder
- For more data, go to: https://www.kaggle.com/competitions/UBC-OCEAN/data

## How to run CNN

## How to run Densenet
1. Go to the **trainDensenet.py** file
2. Replace the value of the variable **dataset_path** with the path of where the **train_thumbnails** folder can be found on your local machine
3. Replace the value of the variable **path** with the path of where the pretrained **DenseNetModel1.pt**, **DenseNetModel2.pt**, **DenseNetModel3.pt**, **DenseNetModel4.pt**, **DenseNetModel5.pt** files can be found on your local machine (All the .pt files should be in the **DenseNetModel** folder)
4. Run the **trainDensenet.py** file
