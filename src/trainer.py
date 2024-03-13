"""
Training script for the UNet model.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class ModelTrainer:
    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss_fn,
        optimizer,
        device,
        batch_size=3,
        num_epochs=10,
        num_workers=4,
    ):
        # Data Loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Model, loss function, optimizer
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Device
        self.device = device

        # Training parameters
        self.num_epochs = num_epochs

    def train_one_epoch(self, epoch_index):
        self.model.train()
        train_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1}")

        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = train_loss / len(self.train_loader)
        print(f"Epoch {epoch_index+1}, Average Loss: {avg_loss:.4f}")

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            # Here you can also call a method to evaluate your model on the test set

    def save_model(self, filename, file_path="./src/models"):
        """
        Saves the model's state dictionary to a file.

        Parameters
        ----------
        filename : str
            Name of the file
        file_path : str, optional
            Location of where to save file, default is 'models' directory

        """
        # Create directory if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path = f"{file_path}/{filename}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")
