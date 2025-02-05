import os
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def save_dataset_json(root_folder='dataset/mvtec', json_path='dataset/dataset.json'):
    """Scans the dataset directory and saves image paths with class labels in a JSON file."""
    dataset_dict = {'train': [], 'val': []}
    classes = [i for i in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, i))]

    all_items = []
    for class_label, target_class in enumerate(classes):
        class_dir = os.path.join(root_folder, target_class, "train", "good")
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img_path = img_path.replace("\\", "/")  # Convert to forward slashes
                all_items.append({"path": img_path, "label": class_label})

    # Shuffle and split dataset
    random.seed(42)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * 0.75)  # 75% for training, 25% for validation
    dataset_dict['train'] = all_items[:split_idx]
    dataset_dict['val'] = all_items[split_idx:]

    # Save to JSON
    with open(json_path, 'w') as file:
        json.dump(dataset_dict, file, indent=4)

class MvTecDataset(Dataset):
    def __init__(self, json_path='dataset/dataset.json', train=True, img_size=256,
                 mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
        """Loads image paths and labels from a JSON file."""
        with open(json_path, 'r') as file:
            dataset_dict = json.load(file)
        
        self.items = dataset_dict['train'] if train else dataset_dict['val']
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size) if train else transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path, class_label = item["path"], item["label"]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, class_label

def create_dataloaders(json_path='dataset/dataset.json', batch_size=32, img_size=256):
    """Creates DataLoaders for training and validation datasets."""
    train_dataset = MvTecDataset(json_path=json_path, train=True, img_size=img_size)
    val_dataset = MvTecDataset(json_path=json_path, train=False, img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
