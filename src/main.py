"""
Script for running the UNet model on the pre-processed DICOM files
"""

import torch
import torch.nn as nn


from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import get_data, get_model_loss
from data import split_data, DICOMSliceDataset
from unet import SimpleUNet
from trainer import ModelTrainer, ModelTrainer_HF_accelerated
from custom_loss import CombinedLoss


def train_save_model(
    loss_fn, num_workers, batch_size=3, num_epochs=10, learning_rate=0.1
):
    """
    Main function to train the model and save the model and metrics.

    Parameters
    ----------
    loss_fn : torch.nn.Module
        The loss function to be used for the segmentation
    num_workers : int
        The number of workers to use for the DataLoader (reccomended to be 4 or 8)
    batch_size : int, optional
        The batch size to use for training (default is set in CW to be 3)
    num_epochs : int, optional
        The number of epochs to train the model for (default is set in CW to be 3)
    learning_rate : float, optional
        The learning rate to use for the model (default is set in CW to be 0.1)
    """
    # Load the pre-processed DICOM data
    data_dict, segmentation_dict = get_data()

    # Split the data into training and test sets
    train_dataset, test_dataset = split_data(data_dict, segmentation_dict, verbose=True)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=3, shuffle=False, num_workers=num_workers
    )

    # Model
    model = SimpleUNet(in_channels=1, out_channels=1)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Trainer
    trainer = ModelTrainer(
        train_dataset,
        test_dataset,
        model,
        loss_fn,
        optimizer,
        device,
        batch_size=3,
        num_epochs=10,
        num_workers=4,
    )

    # Check if model exists
    if trainer.model_exists():
        print("Trained model and corresponding losses found. Loading model...")
        # Load the model
        trainer.load_model()
        # Load the metrics
        metrics = trainer.load_metrics()

    else:
        print("Model does not exist. Training model...")
        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model()

        # Save the metrics
        metrics = trainer.save_metrics()

    # Return trained model and metrics
    return trainer.model, metrics


if __name__ == "__main__":

    args = get_model_loss()
    default_loss = args.default_loss

    if default_loss:
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using default loss function")
    else:
        loss_fn = CombinedLoss()
        print("Using custom loss function")

    # Run the model and save it
    model, metrics = train_save_model(loss_fn, 4)
    print(metrics)
