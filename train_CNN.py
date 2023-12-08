# Train a simple CNN to test our data on
import datasplit
import CNN
import test
import train_
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score

# Load in data
dataset_path = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 184A\\train"
#dataset_path = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 184A\\all_train_data"
# Split data
splits = datasplit.split_data(dataset_path)

indices = [1,]

accuracies = []
precisions = []
confusions = []

for i in range(5):
    # Choose one split to be test data, rest to be train data
    test_loader, train_loader = datasplit.data_loaders(splits[i], splits[:i]+splits[i+1:])

    # Initialize model
    CNNmodel = CNN.model()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(CNNmodel.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1

    # Train model over all epochs
    train.train(CNNmodel,optimizer,criterion,train_loader,num_epochs)

    # Test model
    CNNmodel.eval()
    print(f'Split {i+1}:')
    accuracy, precision, confusion = test.test(CNNmodel, test_loader)
    accuracies.append(accuracy)
    precisions.append(precision)
    confusions.append(confusion)

    # Save model
    model_name = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 184A\\CNNmodel_" + str(i)
    torch.save(CNNmodel, model_name)

print(f'Average accuracy:\t{np.mean(accuracies)}')
print(f'Average precision:\t{np.mean(precisions)}')