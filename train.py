import torch
from training import Trainer 
from models import ResNetSimCLR
from dataset import ContrastiveLearningDataset
from torch.optim.lr_scheduler import StepLR

def main():
    data_loader = ContrastiveLearningDataset("./dataset/data").get_dataset()
    model = ResNetSimCLR(50, device = "cuda", embedding_dim=64)
    lr = 0.01
    weight_decay = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
    trainer = Trainer(model = model, dataloader = data_loader, optimizer = optimizer, save_checkpoints = 20, scheduler = scheduler)
    trainer.train(100)

if __name__ == "__main__":
    main()
