# UBC-Ovarian-Cancer-subtypE-Classification
## Topic overview
https://www.kaggle.com/competitions/UBC-OCEAN/overview

## How to get the data 
### Google Drive
- Download the **train_thumbnails** folder from **https://drive.google.com/drive/folders/1aYZKUWF-ZTx0GpEr1JFvcgmGLEkluh1U?usp=share_link** (Make sure you're logged in using UCI email) This folder contains all the thumbnails images (split into 5 folders) that are used for running the model
- Access pre-trained CNN models here **https://drive.google.com/drive/folders/1MjbilZ4mO8C71ggOB1m8nf9MAvGidnbU?usp=sharing**
- Access pre-trained DenseNet models here **https://drive.google.com/drive/folders/1JJi3Q7VAx5wxR6vugCz3pN7L9Edu6Fek?usp=sharing** 
### Kaggle
- In case the Google Drive link doesn't work, go to **https://www.kaggle.com/competitions/UBC-OCEAN/data** and download the **train_thumbnails** folder on Kaggle
- Go to the **save_images.py** file
- Replace the value of the variable **df** with the path of where the **train.csv** file can be found on your local machine
- Replace the value of the variable **root** with the path of where the **train_thumbnails** folder can be found on your local machine
- Replace the value of the variable **new_root** with the path of where the **train_thumbnails** folder can be found on your local machine
- Run the **save_images.py** to split the data into 5 different folders inside **train_thumbnails**
  
For more information on the dataset, go to: https://www.kaggle.com/competitions/UBC-OCEAN/data

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
3. Replace the value of the variable **path** with the path of where the pretrained files **DenseNetModel1.pt**, **DenseNetModel2.pt**, **DenseNetModel3.pt**, **DenseNetModel4.pt**, **DenseNetModel5.pt** can be found on your local machine (All the .pt files should be in the **Densenet Models** folder)
4. Run the **trainDensenet.py** file
