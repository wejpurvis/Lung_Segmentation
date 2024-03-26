"""
Defines custom PyTorch datasets for the DICOM data
"""

import torch
from torch.utils.data import Dataset


def split_data(data_dict, segmentation_dict, verbose=False, split_ratio=0.67):
    """
    Splits the data and segmentation dictionaries into training and test sets.
    Default split ratio is 2 thirds (66.7%) training and 1 third (33.3%) testing.

    Parameters
    ----------
    data_dict : dict
        Dictionary of patient IDs to their 3D DICOM data numpy arrays
    segmentation_dict : dict
        Dictionary of patient IDs to their 3D segmentation mask numpy arrays
    verbose : bool, optional
        Whether to print the size of the training and test datasets. Default is False.
    split_ratio : float, optional
        The ratio of training data to validation data. Default is 0.67.

    Returns
    -------
    tuple
        A tuple containing the training and test datasets
    """
    # Split the data into training and validation sets
    num_train = int(len(data_dict) * split_ratio)
    num_test = len(data_dict) - num_train

    train_ids = list(data_dict.keys())[:num_train]
    test_ids = list(data_dict.keys())[num_train:]

    train_data_dict = {k: data_dict[k] for k in train_ids}
    test_data_dict = {k: data_dict[k] for k in test_ids}

    train_segmentation_dict = {k: segmentation_dict[k] for k in train_ids}
    test_segmentation_dict = {k: segmentation_dict[k] for k in test_ids}

    # Initialise the datasets
    train_dataset = DICOMSliceDataset(train_data_dict, train_segmentation_dict)
    test_dataset = DICOMSliceDataset(test_data_dict, test_segmentation_dict)

    if verbose:
        print(f"Splitting data into training and test sets")
        print(
            f"Training dataset has {num_train} cases, with {len(train_dataset)} slices"
        )
        print(f"Test dataset has {num_test} cases, with {len(test_dataset)} slices")
    return train_dataset, test_dataset


class DICOMSliceDataset(Dataset):
    """
    Custom PyTorch dataset for the DICOM and segmentation data.
    Defines the __len__ and __getitem__ methods to retrieve slices and masks, as required by PyTorch.

    Methods
    -------
    __init__(data_dict, segmentation_dict, transform=None)
        Initializes the dataset with the DICOM and segmentation data.
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Retrieves the slice and mask at the given index.

    """

    def __init__(self, data_dict, segmentation_dict, transform=None):
        """
        Parameters
        ----------
        data_dict : dict
            Dictionary of patient IDs to their 3D DICOM data numpy arrays
        segmentation_dict : dict
            Dictionary of patient IDs to their 3D segmentation mask numpy arrays
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
        self.slices = []
        self.masks = []
        self.transform = transform

        # Flatten 3D data into 2D slices
        for patient_id, data in data_dict.items():
            num_slices = data.shape[0]
            for slice_idx in range(num_slices):
                self.slices.append(data[slice_idx, :, :])
                self.masks.append(segmentation_dict[patient_id][slice_idx, :, :])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.slices)

    def __getitem__(self, idx):
        """
        Retrieves the slice and mask at the given index.

        Parameters
        ----------
        idx : int
            Index of the slice and mask to retrieve.

        Returns
        -------
        torch.Tensor
            A tensor representing the slice.
        """
        slice = self.slices[idx]
        mask = self.masks[idx]

        if self.transform:
            slice = self.transform(slice)
            mask = self.transform(mask)

        # Convert numpy arrays to torch tensors
        slice = torch.from_numpy(slice).float()
        mask = torch.from_numpy(mask).float()

        # Add a channel dimension to the slice [1, y, z]
        slice = slice.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return slice, mask
