"""
Training script for the UNet model.
"""

import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import os
import time
import json


class ModelTrainer:
    """
    A simplified training class for a PyTorch model focusing on the training process. It includes basic functionalities
    for training a model, with optional extensions for model evaluation and saving.

    Parameters
    ----------
    train_dataset : Dataset
        The dataset for training the model.
    test_dataset : Dataset
        The dataset for testing the model, if evaluation is performed manually.
    model : torch.nn.Module
        The model to be trained.
    loss_fn : Callable
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    device : torch.device
        The device on which to train the model (e.g., 'cpu' or 'cuda').
    batch_size : int, optional
        The size of each batch of data. Default is 3.
    num_epochs : int, optional
        The number of epochs for training. Default is 10.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 4.

    Methods
    -------
    train_one_epoch(epoch_index)
        Trains the model for one epoch and logs the progress.
    train()
        Conducts the training process over the specified number of epochs.
    save_model(filename, file_path="./src/models")
        Saves the model state to a specified path, facilitating model reusability.
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

        # Metrics
        self.accuracy_metric = torchmetrics.BinaryAccuracy(threshold=0.5).to(device)
        self.train_losses = []
        self.train_accuracies = []

    def train_one_epoch(self, epoch_index):
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
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = train_loss / len(self.train_loader)
        print(f"Epoch {epoch_index + 1}, Average Loss: {avg_loss:.4f}")

    def train(self):
        print("Training the model...")
        start_time = time.time()
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            # Here you can also call a method to evaluate your model on the test set
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds")

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


class ModelTrainer2:
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


    Methods
    -------
    train_one_epoch(epoch_index)
        Trains the model for one epoch and logs the progress.
    evaluate()
        Evaluates the model on the test dataset.
    train()
        Conducts the training process over the specified number of epochs, including evaluation.
    save_model(filename, file_path="./models")
        Saves the model state to a specified path.
    save_metrics(filename, file_path="./models")
        Saves the training and testing metrics to a specified path.
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
            for images, masks in self.test_loader:
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
        print("Training the model...")
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

    def save_metrics(self, file_path="./models"):
        """
        Save the model hyperparameters and metrics to a JSON file.
        The training losses and accuracies, as well as the test losses and accuracies, are saved as dictionaries to a .JSON file.

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
        # Save to a JSON file
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Model parameters saved at {path}")
