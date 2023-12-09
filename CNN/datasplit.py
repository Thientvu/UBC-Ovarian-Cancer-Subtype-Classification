import os 
import random
from sklearn.model_selection import KFold
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def process_data(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((900,900)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    return dataset

def split_data(dataset_path):
    dataset = process_data(dataset_path)

    # Create split sizes
    split_sizes = [int(len(dataset) * 0.2) for _ in range(5)]
    # Add remaining data to first split
    split_sizes[0] += len(dataset)%(int(len(dataset)*0.2)*5)
    
    splits = random_split(dataset, split_sizes)

    return splits

def data_loaders(test_dataset, train_splits):
    # Creat test loader
    test_loader =  torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True)

    # Combine train splits into one dataset
    train_dataset = torch.utils.data.ConcatDataset(train_splits)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True)

    return test_loader, train_loader    
