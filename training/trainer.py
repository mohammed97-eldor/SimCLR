import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from training.utils import contrastive_loss, l2_normalize
from dataset import preprocess_for_train

class Trainer:
    def __init__(self, model, dataloader, optimizer, save_checkpoints = None,
                 criterion = contrastive_loss, normalizer = l2_normalize,
                 checkpoint_dir = "./checkpoints", scheduler=None):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.total_loss = []
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        self.tracker = {}
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "total_loss": self.total_loss,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # first_layer_name = list(self.model.state_dict().keys())[0]  # Name of the first layer
        # first_layer_weights = self.model.state_dict()[first_layer_name]  # Weights of the first layer
        # print(f"First layer name: {first_layer_name}")
        # print("First layer weights:", first_layer_weights[0,:,:,:])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # first_layer_name = list(self.model.state_dict().keys())[0]  # Name of the first layer
        # first_layer_weights = self.model.state_dict()[first_layer_name]  # Weights of the first layer
        # print(f"First layer name: {first_layer_name}")
        # print("First layer weights:", first_layer_weights[0,:,:,:])

        # print(self.optimizer)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # print(self.optimizer)
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.total_loss = checkpoint.get("total_loss", [])
        print(f"Checkpoint loaded from: {checkpoint_path}")
        
        # Return the epoch to resume from
        return checkpoint["epoch"]

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = []
        for mini_batch in tqdm(self.dataloader):
            mini_batch = mini_batch[0].to(self.device)
            augmented_batch = []  # To store 2N images
            for image in mini_batch:
                augmented_1 = preprocess_for_train(image, height=32, width=32, color_distort=True, crop=True, flip=True)  # First augmentation
                augmented_2 = preprocess_for_train(image, height=32, width=32, color_distort=True, crop=True, flip=True)  # Second augmentation
                augmented_batch.extend([augmented_1, augmented_2])  # Append both

            augmented_batch = torch.stack(augmented_batch)  # Shape: [2N, C, H, W]
            representations = self.model(augmented_batch.permute(0,3,1,2))
            projections = self.normalizer(representations, dim=1)
            loss = self.criterion(projections)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
        
        if self.scheduler:
            self.scheduler.step()

        self.tracker[epoch+1] = {"Loss": np.mean(losses), "lr": self.optimizer.param_groups[0]['lr']}
        
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses)}, lr: {self.optimizer.param_groups[0]['lr']}")

        return losses

    def train(self, num_epochs, resume_checkpoint=None):
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.load_checkpoint(resume_checkpoint)
        for epoch in range(start_epoch, num_epochs):
            losses = self.train_one_epoch(epoch)
            self.total_loss.extend(losses)

            if self.save_checkpoints and (epoch+1)%self.save_checkpoints == 0:
                self.save_checkpoint(epoch)
        
        tracker_path = os.path.join(self.checkpoint_dir, "tracker.json")
        with open(tracker_path, "w") as f:
            json.dump(self.tracker, f)
        print(f"Tracker saved to {tracker_path}")

        # Plot loss as a function of epoch
        epochs = list(self.tracker.keys())
        losses = [self.tracker[epoch]["Loss"] for epoch in epochs]

        plt.figure()
        plt.plot(epochs, losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.legend()
        plt.grid()
        loss_plot_path = os.path.join(self.checkpoint_dir, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")

if __name__ == "__main__":
    print("The trainer class is a custom class and to run the training script refer to the main function")
