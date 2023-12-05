# Train a simple CNN to test our data on
import datasplit
import CNN
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score

# Load in data
dataset_path = "/Users/thientoanvu/Desktop/Classes/CS184A/Final-project/train_thumbnails"
# Split data
splits = datasplit.split_data(dataset_path)
# Choose one split to be test data, rest to be train data
test_loader, train_loader = datasplit.data_loaders(splits[0], splits[1:])

# Initialize model

CNNmodel = CNN.Model()
learning_rate = 0.001
optimizer = torch.optim.SGD(CNNmodel.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

num_epochs = 1

# Train model
total_step = len(train_loader)
for epoch in range(num_epochs):
    CNNmodel.train()
    for i, data in enumerate(train_loader):
        image, label = data
        label = torch.eye(6)[label]

        # Forward pass
        output = CNNmodel(image)
        loss = criterion(output,label)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    

# Test model
CNNmodel.eval()
y_pred_list = np.array([])
y_target_list=np.array([])
with torch.no_grad():
    for i, data in enumerate(test_loader):
        image, label = data
        output = CNNmodel(image)
        y_pred_tag = torch.argmax(output,1)

        #print(y_pred_tag, label)
        
        y_pred_list= np.append(y_pred_list, y_pred_tag.detach().numpy())
        y_target_list = np.append(y_target_list, label.detach().numpy())

print("accuracy")
print(accuracy_score(y_target_list,y_pred_list))

print("confusion matrix")
print(confusion_matrix(y_target_list,y_pred_list))

print("precision score")
print(precision_score(y_target_list,y_pred_list,
                      average='weighted',
                      zero_division=0))