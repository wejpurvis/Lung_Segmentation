"""
Training script for the UNet model.
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import os
import time
import json


class ModelTrainer:
    """
    A training class for the UNet model which includes functionality for both training and evaluation.

    Supports saving model and model parameters.

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        The training dataset
    test_dataset : torch.utils.data.Dataset
        The test dataset
    model : torch.nn.Module
        The model to train (UNet)
    loss_fn : torch.nn.Module
        The loss function
    optimizer : torch.optim.Optimizer
        The optimizer
    device : torch.device
        The device to train the model on
    batch_size : int, optional
        The batch size, default is 3
    num_epochs : int, optional
        The number of epochs, default is 10
    num_workers : int, optional
        The number of workers for the DataLoader, default is 4
    """

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
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.set_model_name(loss_fn.__class__.__name__, batch_size, optimizer)
        print(f"Model name set to: {self.model_name}")

    def train_one_epoch(self, epoch_index):
        """
        Train the model for one epoch and calculates the average loss and accuracy for that epoch.

        Parameters
        ----------
        epoch_index : int
            The index of the current epoch

        """
        self.model.train()
        self.accuracy_metric.reset()
        train_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_index + 1}")

        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(outputs)
            self.accuracy_metric.update(preds, masks.int())

        avg_loss = train_loss / len(self.train_loader)
        avg_accuracy = self.accuracy_metric.compute()
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy.item())

        print(
            f"Epoch {epoch_index + 1}, Average Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}"
        )

    def evaluate(self):
        self.model.eval()
        self.accuracy_metric.reset()
        test_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Evaluation")
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device).float()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

                test_loss += loss.item()
                preds = torch.sigmoid(outputs)
                self.accuracy_metric.update(preds, masks.int())

        avg_loss = test_loss / len(self.test_loader)
        avg_accuracy = self.accuracy_metric.compute()
        self.test_losses.append(avg_loss)
        self.test_accuracies.append(avg_accuracy.item())

        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    def train(self):
        """
        Primary training function for the `ModelTrainer` class.
        Uses the `train_one_epoch` and `evaluate` functions to train the model over the specified number of epochs and to subsequently evaluate the model after each epoch using BinarryAccuracy and the specified loss function.
        """
        print(f"\nTraining the model:")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.evaluate()  # Evaluate on test data after each epoch
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds")

    def set_model_name(self, loss_fn, batch_size, optimizer):
        try:
            learning_rate = optimizer.param_groups[0]["lr"]
        except:
            learning_rate = "lr"
        self.model_name = f"unet_{loss_fn}_bs{batch_size}_lr{learning_rate}"

    def save_model(self, file_path="./models"):
        """
        Save the trained model as a .pth file.

        Parameters
        ----------
        file_path : str, optional
            The location of where to save the file, default is 'models' directory
        """
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        path = os.path.join(file_path, f"{filename}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")

    def model_exists(self, file_path="./models"):
        """
        Check if the model file and its metrics exists.

        Parameters
        ----------
        file_path : str, optional
            The location of where to save the file, default is 'models' directory

        Returns
        -------
        bool
            True if the model file exists, False otherwise
        """
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        model_path = os.path.join(file_path, f"{filename}.pth")
        metrics_path = os.path.join(file_path, f"{filename}.csv")

        return os.path.exists(model_path) and os.path.exists(metrics_path)

    def load_model(self, file_path="./models"):
        """
        Load the trained model from a .pth file.

        Parameters
        ----------
        file_path : str, optional
            The location of where to load the file from, default is 'models' directory
        """
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        path = os.path.join(file_path, f"{filename}.pth")
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return self.model

    def load_metrics(self, file_path="./models"):
        """
        Load the model losses and accuracies from a .CSV file.

        Parameters
        ----------
        file_path : str, optional
            The location of where to load the file from, default is 'models' directory
        """
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        path = os.path.join(file_path, f"{filename}.csv")
        with open(path, "r") as f:
            metrics = json.load(f)
        self.train_losses = metrics["train_losses"]
        self.train_accuracies = metrics["train_accuracies"]
        self.test_losses = metrics["test_losses"]
        self.test_accuracies = metrics["test_accuracies"]
        print(f"Model parameters loaded from {path}")
        return metrics

    def save_metrics(self, file_path="./models"):
        """
        Save the model losses and accuracies to a .CSV file.
        The training losses and accuracies, as well as the test losses and accuracies, are saved as dictionaries to a .CSV file.

        Parameters
        ----------
        file_path : str, optional
            The location of where to save the file, default is 'models' directory
        """
        if not hasattr(self, "model_name"):
            print("Model name not set")
            filename = "unet_model"
        else:
            filename = self.model_name

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        path = os.path.join(file_path, f"{filename}.csv")
        metrics = {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "test_losses": self.test_losses,
            "test_accuracies": self.test_accuracies,
        }
        # Save to a CSV file
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Model parameters saved at {path}")
        return metrics
