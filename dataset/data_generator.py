import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class ContrastiveLearningDataset:

    def __init__(self, root_folder = './data', batch_size = 32, img_size = 32, train_sample = 1000, val_sample = 200):
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_sample = train_sample
        self.val_sample = val_sample

    def get_transformations(self, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):

        transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        
        return transform
    
    def get_dataset(self, transform=None):

        if transform == None:
            transform = self.get_transformations()

        dataset = CIFAR10(root=self.root_folder, train=True, download=True, transform=transform)

        subset_indices = list(range(self.train_sample))
        small_dataset = torch.utils.data.Subset(dataset, subset_indices)

        # DataLoader for batching
        data_loader = DataLoader(small_dataset, batch_size=self.batch_size, shuffle=True)

        return data_loader
    
    def get_dataset_val(self, transform=None):

        if transform == None:
            transform = self.get_transformations()

        dataset = CIFAR10(root=self.root_folder, train=True, download=True, transform=transform)

        subset_indices = list(range(self.train_sample, self.train_sample+self.val_sample+1))
        small_dataset = torch.utils.data.Subset(dataset, subset_indices)

        # DataLoader for batching
        data_loader = DataLoader(small_dataset, batch_size=self.batch_size, shuffle=True)

        return data_loader
    
if __name__ == "__main__":
    print(ContrastiveLearningDataset().get_dataset())
    print(ContrastiveLearningDataset().get_dataset_val())
