"""
Defines custom PyTorch datasets for the DICOM data
"""

import torch
from torch.utils.data import Dataset


# TODO: check if data is normalised


class DICOMSliceDataset(Dataset):
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
        return len(self.slices)

    def __getitem__(self, idx):
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
