"""
Script for running the UNet model on the pre-processed DICOM files
"""

import torch

from utils import get_data
from data import split_data, DICOMSliceDataset
from unet import SimpleUNet
from trainer import ModelTrainer


if __name__ == "__main__":
    # Load the pre-processed DICOM data
    data_dict, segmentation_dict = get_data()

    # Split the data into training and test sets
    train_dataset, test_dataset = split_data(data_dict, segmentation_dict, verbose=True)

    # Model
    model = SimpleUNet(in_channels=1, out_channels=1)

    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

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
        num_workers=8,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("unet_model_not_custom")
