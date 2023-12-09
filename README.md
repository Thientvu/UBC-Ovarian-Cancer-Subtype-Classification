# UBC-Ovarian-Cancer-subtypE-Classification
## Topic overview
https://www.kaggle.com/competitions/UBC-OCEAN/overview

## Dataset
- The thumbnails images that are separated into 5 different classes can be found in the **train_thumbnails** folder
- For more data, go to: https://www.kaggle.com/competitions/UBC-OCEAN/data

## How to train a CNN
1. Go to **train_CNN.py**.
2. In **main**, replace **dataset_path** with the path to your dataset.
3. In **main**, replace **model_name** with the path to where you would like to save your model.

   train_CNN.py will create and train a CNN model on the given dataset over 5 folds, 5 epochs each. It will print the accuracy, precision, and confusion of each model as it is trained.

## How to run a pre-trained CNN
1. Go to **run_CNN.py**
2. Replace **dataset_path** with the path to your dataset.
3. Replace **model** with the path to your pre-trained model.

   run_CNN.py will run a pre-trained model on the given dataset and print the accuracy, precision, and confusion of the model.

## How to run Densenet
1. Go to the **Densenet** folder, and click on **trainDensenet.py** 
2. Replace the value of the variable **dataset_path** with the path of where the **train_thumbnails** folder can be found on your local machine
3. Replace the value of the variable **path** with the path of where the pretrained **DenseNetModel1.pt**, **DenseNetModel2.pt**, **DenseNetModel3.pt**, **DenseNetModel4.pt**, **DenseNetModel5.pt** files can be found on your local machine (All the .pt files should be in the **DenseNetModel** folder)
4. Run the **trainDensenet.py** file
