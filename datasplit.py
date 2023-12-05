import os 
import random
from sklearn.model_selection import KFold
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def process_data(dataset_path):
    #Apply data augmentation techniques to artificially increase the diversity of your training set. 
    #This can help the model generalize better to unseen data.
    transform = transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    return dataset

def split_data(dataset_path, n_splits=5):
    dataset = process_data(dataset_path)

    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    splits = []

    for train_index, test_index in kf.split(dataset):
        train_set = Subset(dataset, train_index)
        test_set = Subset(dataset, test_index)
        splits.append((train_set, test_set))

    return splits

def data_loaders(test_dataset, train_splits):
    # Create test loader
    test_loader =  torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True)

    # Combine train splits into one dataset
    train_dataset = torch.utils.data.ConcatDataset(train_splits)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True)

    return test_loader, train_loader    
