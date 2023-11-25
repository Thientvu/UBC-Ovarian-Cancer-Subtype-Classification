import os 
import random
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

dataset_path = "The path for the training data set"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
  
split_sizes = [int(len(dataset) * 0.2) for _ in range(5)]
  
splits = random_split(dataset, split_sizes)
  
split_1, split_2, split_3, split_4, split_5 = splits
loader_split_1 = DataLoader(split_1, batch_size=32, shuffle=True)
loader_split_2 = DataLoader(split_2, batch_size=32, shuffle=True)
loader_split_3 = DataLoader(split_3, batch_size=32, shuffle=True)
loader_split_4 = DataLoader(split_4, batch_size=32, shuffle=True)
loader_split_5 = DataLoader(split_5, batch_size=32, shuffle=True)
