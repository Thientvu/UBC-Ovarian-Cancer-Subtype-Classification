import datasplitDensenet
import Densenet
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score

def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    print("Training the model")
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            image, label = data
            label = torch.eye(6)[label]

            # Forward pass
            output = model(image)
            loss = criterion(output, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 41 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

def evaluate_model(model, test_loader):
    print("Evaluating the model")

    model.eval()
    y_pred_list = np.array([])
    y_target_list = np.array([])
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image, label = data
            output = model(image)
            y_pred_tag = torch.argmax(output, 1)

            y_pred_list = np.append(y_pred_list, y_pred_tag.detach().numpy())
            y_target_list = np.append(y_target_list, label.detach().numpy())

    accuracy = accuracy_score(y_target_list, y_pred_list)
    confusion_mat = confusion_matrix(y_target_list, y_pred_list)
    precision = precision_score(y_target_list, y_pred_list, average='weighted', zero_division=0)

    return accuracy, confusion_mat, precision

def main():
    # Load in data
    dataset_path = "/Users/thientoanvu/Desktop/Classes/CS184A/Final-project/train_thumbnails"

    # Split data into 5 folds
    splits = datasplitDensenet.split_data(dataset_path)

    # Initialize model
    learning_rate = 0.001

    num_epochs = 5

    # Perform 5-fold cross-validation
    for fold, (train_set, test_set) in enumerate(splits):
        print(f"Fold {fold + 1}:")

        path = f"/Users/thientoanvu/Desktop/Classes/CS184A/Final-project/Pretrained-Densenet-Models/DenseNetModel{fold + 1}.pt"
        
        # Initialize a new model for each fold
        #model = Densenet.DenseNet()
        model = torch.load(path)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Set up data loaders
        test_loader, train_loader = datasplitDensenet.data_loaders(test_set, [train_set])

        # Train model
        #train_model(model, train_loader, criterion, optimizer, num_epochs)
        #torch.save(model, path)

        # Evaluate model
        accuracy, confusion_mat, precision = evaluate_model(model, test_loader)
        
        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(confusion_mat)
        print("Precision Score:", precision)
        print("\n")

if __name__ == "__main__":
    main()
