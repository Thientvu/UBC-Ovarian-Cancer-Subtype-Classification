# Train a simple CNN to test our data on
import datasplit
import CNN
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score

def train(model, optimizer, criterion, train_loader,num_epochs=1):
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            image, label = data
            label = torch.eye(6)[label]

            # Forward pass
            output = model(image)
            loss = criterion(output,label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

def test(model, data_loader):
    y_pred_list = np.array([])
    y_target_list=np.array([])
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            image, label = data
            output = model(image)
            y_pred_tag = torch.argmax(output,1)
            
            y_pred_list= np.append(y_pred_list, y_pred_tag.detach().numpy())
            y_target_list = np.append(y_target_list, label.detach().numpy())

    accuracy = accuracy_score(y_target_list,y_pred_list)    

    precision = precision_score(y_target_list,y_pred_list,
                        average='weighted',
                        zero_division=0)
    
    confusion = confusion_matrix(y_target_list,y_pred_list)

    print(f"Accuracy:\t{accuracy}")

    print(f"Precision score:\t{precision}")

    print(f"Confusion matrix:\n{confusion}")
    
    return accuracy, precision, confusion

def main():
    # Load in data
    
    dataset_path = "data path name"
    # Split data
    splits = datasplit.split_data(dataset_path)

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

        num_epochs = 5

        # Train model over all epochs
        train(CNNmodel,optimizer,criterion,train_loader,num_epochs)

        # Test model
        CNNmodel.eval()
        print(f'Split {i+1}:')
        accuracy, precision, confusion = test(CNNmodel, test_loader)
        accuracies.append(accuracy)
        precisions.append(precision)
        confusions.append(confusion)

        # Save model
        model_name = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 184A\\Models\\CNNmodel_full_" + str(i) + '.pt'
        torch.save(CNNmodel, model_name)

    print(f'Average accuracy:\t{np.mean(accuracies)}')
    print(f'Average precision:\t{np.mean(precisions)}')