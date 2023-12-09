import datasplit
from test import test
from train_ import train
import torch
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score

# Load data
dataset_path = "path to dataset"
data = datasplit.process_data(dataset_path)
data_loader = torch.utils.data.ConcatDataset(data)

# Load model
model = torch.load("path to CNN model")

# Print metrics
accuracy, precision, confusion = test(model, data_loader)

print(f"Accuracy:\t{accuracy}")

print(f"Precision score:\t{precision}")

print(f"Confusion matrix:\n{confusion}")

