import os
import torch
from training import Trainer
from models import ResNetSimCLR
from dataset import save_dataset_json, MvTecDataset, create_dataloaders
from torch.optim.lr_scheduler import StepLR
import argparse

def main(data_dir="./dataset/mvtech",
         json_path = "./dataset/dataset.json",
         resnet_size=50, 
         device="cuda", 
         embedding_dim=64, 
         lr=0.01, 
         weight_decay=0.0001, 
         step_size=5, 
         gamma=0.5, 
         save_checkpoints=5, 
         num_epochs=40, 
         resume_checkpoint=None,
         optimizer_type = "SGD",
         validationstep = 5,
         use_scheduler = False):

    save_dataset_json(data_dir, json_path)
    data_loader, data_loader_val = create_dataloaders(json_path=json_path)
    
    # Initialize model
    model = ResNetSimCLR(size=resnet_size, device=device, embedding_dim=embedding_dim)
    
    # Set up optimizer and scheduler
    optimizer_dict = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW
    }
    optimizer = optimizer_dict[optimizer_type](model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        dataloader=data_loader,
        optimizer=optimizer,
        save_checkpoints=save_checkpoints,
        validationstep = validationstep,
        scheduler=scheduler,
        dataloader_val = data_loader_val,
        device = device
    )
    
    # Start or resume training
    trainer.train(num_epochs, resume_checkpoint=resume_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SimCLR model.")
    parser.add_argument("--data_dir", type=str, default="./dataset/mvtec", help="Path to dataset directory.")
    parser.add_argument("--resnet_size", type=int, default=50, choices=[18, 50, 101], help="Size of ResNet model.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimensionality of the embedding space.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer.")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for LR scheduler.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma value for LR scheduler.")
    parser.add_argument("--save_checkpoints", type=int, default=5, help="Frequency (in epochs) to save checkpoints. Set to None to disable.")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--optimizer_type", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="Training optimizer")
    parser.add_argument("--validationstep", type=int, default=5, help="Apply validation every validationstep")
    parser.add_argument("--use_scheduler", type=bool, default=True, help="True for learning rate schedualer")

    args = parser.parse_args()
    main(**vars(args))
