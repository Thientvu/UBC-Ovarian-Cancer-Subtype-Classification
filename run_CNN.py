import datasplit
from train_CNN import test
import torch
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score

# Load data
dataset_path = "C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 184A\\all_train_data"
data = datasplit.process_data(dataset_path)
data_loader =  torch.utils.data.DataLoader(dataset=data, shuffle=True)

# Load model
model = torch.load("C:\\Users\\blues\\OneDrive\\Desktop\\UCI\\Fall '23\\CS 184A\\Models\\CNNmodel_full_2.pt")

# Print metrics
accuracy, precision, confusion = test(model, data_loader)

print(f"Accuracy:\t{accuracy}")

print(f"Precision score:\t{precision}")

print(f"Confusion matrix:\n{confusion}")

