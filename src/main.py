import os
import pickle
from plotting import plot_patients
from utils import get_data


# Load DICOM and segmentation data for patients
dicom_dict, segmentation_dict = get_data()
plot_patients(dicom_dict, segmentation_dict, overlay=True, save=True, animate=False)
