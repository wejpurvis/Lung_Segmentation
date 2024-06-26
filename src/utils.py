"""
Utility functions for Medical Imaging project
"""

import os
import torch
import argparse
import pickle
import numpy as np
import pydicom
from tqdm import tqdm
import concurrent.futures


def load_patient_dicom(patient_path: str, anonymize: bool = True) -> np.ndarray:
    """
    Converts all DICOM files for a particular patient into a 3D numpy array

    Parameters
    ----------
    patient_path : str
        Path to the patient's DICOM files
    anonymize : bool, optional
        Whether to anonymize the patient's data (default is False)

    Returns
    -------
    tuple
        The patient's case ID and 3D numpy array of the DICOM files. The array has shape
        [x, y, z] where x is the number of slices (axial plane), y is the
        coronal plane, and z is the sagittal plane
    """

    # Store the slice location and pixel data for each DICOM file
    dicom_data_and_location = []

    for f in os.listdir(patient_path):
        full_path = os.path.join(patient_path, f)
        if f.endswith(".dcm"):
            try:
                case_id = patient_path.split("/")[-1]
                dicom_data = pydicom.dcmread(full_path)

                # Anonymize patient data (if requested)
                if anonymize:
                    dicom_data.PatientName = case_id
                    dicom_data.PatientID = case_id
                    dicom_data.PatientBirthDate = ""
                    try:
                        del dicom_data.PatientBirthTime
                    except:
                        pass

                # Rescale pixel data (if provided)
                rescale_slope = getattr(dicom_data, "RescaleSlope", 1)
                rescale_intercept = getattr(dicom_data, "RescaleIntercept", 0)
                pixel_data = dicom_data.pixel_array
                pixel_data = pixel_data * rescale_slope + rescale_intercept

                slice_location = dicom_data.SliceLocation

                if pixel_data.shape == (512, 512):
                    dicom_data_and_location.append((slice_location, pixel_data))
                else:
                    print(
                        f"Warning: Image {f} skipped due to "
                        f"unexpected image dimensions {pixel_data.shape}"
                    )
            except AttributeError as e:
                print(f"Warning: Image {f} skipped due to missing attribute: {e}")
                continue

    # Sort list of pixel data and slice locations based on slice location
    dicom_data_and_location.sort(key=lambda x: x[0], reverse=True)

    num_slices = len(dicom_data_and_location)

    # Initialize empty 3d numpy array (assumes all images are 512 by 512)
    sorted_dicom_data = np.zeros((num_slices, 512, 512))

    # Load each DICOM file into the 3d numpy array
    for i, (_, pixel_data) in enumerate(dicom_data_and_location):
        sorted_dicom_data[i, :, :] = pixel_data

    return (case_id, sorted_dicom_data)


def load_patient_segmentations(segmentations_path: str, loading: bool = False) -> dict:
    """
    Load segmentation data for patients

    Parameters
    ----------
    segmentations_path : str
        Path to the segmentation files for all patients
    loading : bool, optional
        Whether to display a progress bar (default is False)

    Returns
    -------
    dict
        A dictionary of segmentations for each patient
    """
    # Initialise dictionary of case_ids and segmentations
    segmentations_dict = {}

    # Filter .npz files
    segmentation_files = [
        file for file in os.listdir(segmentations_path) if file.endswith(".npz")
    ]

    # Optionally wrap the file list with tqdm for a progress bar
    files_to_process = tqdm(segmentation_files) if loading else segmentation_files

    for segmentation_file in files_to_process:
        # Assuming file format is: 'Case_XXX_seg_npz' where XXX is cased ID
        case_number = segmentation_file.split("_")[1]
        case_id = f"Case_{case_number}"
        segmentation = np.load(os.path.join(segmentations_path, segmentation_file))
        try:
            masks = segmentation["masks"]
            # Invert axial plane of mask (to match images)
            inverted_masks = masks[::-1, :, :]
            # Update hash map
            segmentations_dict[case_id] = inverted_masks
        except KeyError:
            print(f"No masks found for Case {case_number}")
            segmentations_dict[case_id] = []

    return segmentations_dict


def load_all_patient_dicoms_para(
    dataset_path: str, loading: bool = False, anonymize: bool = False
) -> dict:
    """
    Load process DICOM data for patients
    Uses threading to optimise loading of DICOM data

    Parameters
    ----------
    dataset_path : str
        Path to the DICOM files for all patients
        (dataset_path -> patient dir -> dicom files)
    loading : bool, optional
        Whether to display a progress bar (default is False)
    anonymize: bool, optional
        Whether to remove personal identifiers in DICOM files of patients
        (default is False)

    Returns
    -------
    dict
        A dictionary of DICOMS (as 3D numpy arrays) for each patient
    """
    # Initialise dictionary of case_ids and dicoms
    dicoms_dict = {}

    def process_patient(patient):
        """
        Nested function for concurrent execution
        """
        patient_path = os.path.join(dataset_path, patient)
        case_id, dicom_data = load_patient_dicom(patient_path, anonymize)
        return case_id, dicom_data

    # Iterate through patient directories
    patients = [
        patient
        for patient in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, patient))
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create all futures
        futures = [executor.submit(process_patient, patient) for patient in patients]

        # Wrap the concurrent.futures `as_completed` with tqdm for progress tracking
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            disable=not loading,
        ):
            case_id, dicom_data = future.result()
            dicoms_dict[case_id] = dicom_data

    # Sort dicom dict based on keys
    sorted_dicoms_dict = {k: dicoms_dict[k] for k in sorted(dicoms_dict.keys())}

    return sorted_dicoms_dict


def get_data():
    """
    Load DICOM and segmentation data for patients
    If pickle files are present, load them, otherwise load the data from the provided
    DICOM and .npz files, convert them to appropriate data structures and save them to
    pickle files.

    Returns
    -------
    tuple
        A tuple of dictionaries containing DICOM and segmentation data for patients
    """
    data_path = os.path.join(os.getcwd(), "data")

    segs = None
    dics = None

    # Path for pickle files
    seg_pickle_path = os.path.join(data_path, "segmentations.pkl")
    dic_pickle_path = os.path.join(data_path, "dicoms.pkl")

    # TODO: Seperate the loading of DICOM and segmentation data
    # Check if DICOM pickle file (from first part) exists
    if os.path.exists(dic_pickle_path):
        print("Processed DICOM data found, loading from pickle file...")
        with open(dic_pickle_path, "rb") as file:
            dics = pickle.load(file)

    else:
        print("Processed DICOM data not found, loading data from DICOM files...")

        if not os.path.exists(os.path.join(data_path, "Images")):
            raise FileNotFoundError(
                "\n\nDICOM data not found: please download it and place it in the 'data' directory.\n"
            )

        dic_path = os.path.join(data_path, "Images")
        dics = load_all_patient_dicoms_para(dic_path, loading=True)

        # Save the data as a pickle file
        with open(dic_pickle_path, "wb") as file:
            pickle.dump(dics, file)
            print("Compressed DICOM data saved to 'data/dicoms.pkl'")

    # Check if segmentation pickle file exists
    if os.path.exists(seg_pickle_path):
        print("Processed segmentation data found, loading from pickle file...")
        with open(seg_pickle_path, "rb") as file:
            segs = pickle.load(file)

    else:
        print("Processed segmentation data not found, loading data from .npz files...")

        if not os.path.exists(os.path.join(data_path, "Segmentations")):
            raise FileNotFoundError(
                "\n\nSegmentation data not found: please download it and place it in the 'data' directory.\n"
            )

        seg_path = os.path.join(data_path, "Segmentations")
        segs = load_patient_segmentations(seg_path, loading=True)

        # Save the data as a pickle file
        with open(seg_pickle_path, "wb") as file:
            pickle.dump(segs, file)
            print("Compression segmentation data saved to 'data/segmentations.pkl'")

    return dics, segs


def get_dicom_args():
    """
    Get CL arguments for running load_dicom.py (Module 1)

    Returns
    -------
    argparse.Namespace
        The CL arguments (whether to anonymize patient data)
    """

    parser = argparse.ArgumentParser(description="Load DICOM data for Medical Imaging")
    parser.add_argument(
        "--anonymize",
        action="store_false",
        help="Anonymize patient data by removing personal identifiers (default to True)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_false",
        help="Show progress bar when loading DICOM data (default to True)",
    )

    return parser.parse_args()


def get_model_loss():
    """
    Get CL arguments for running model (custom loss or standard BCEWithLogitsLoss)

    Returns
    -------
    argparse.Namespace
        The CL arguments (whether to use custom loss function)
    """

    parser = argparse.ArgumentParser(
        description="Run U-Net model for image segmentation"
    )
    parser.add_argument(
        "--default-loss",
        action="store_true",
        help="Use default loss function for training the model. If declared, BCEWithLogitsLoss will be used as the loss function",
    )

    return parser.parse_args()


def dice_coefficient(prediction, target):
    """
    Calculate the Dice similarity coefficient (DSC) for a given prediction and target mask. (used in `evaluate_slices` function)

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted mask
    target : torch.Tensor
        The target mask

    Returns
    -------
    float
        The Dice similarity coefficient (DSC)
    """
    epsilon = 1e-6
    intersection = (prediction * target).sum()
    dice = (2.0 * intersection + epsilon) / (prediction.sum() + target.sum() + epsilon)
    return dice


def evaluate_slices(model, dataloader, device):
    """
    Evaluate the model after training by calculating the binary accuracy and Dice similarity coefficient (DSC) for each slice for a given dataloader (test or train).

    Parameters
    ----------
    model : torch.nn.Module
        The trained model
    dataloader : torch.utils.data.DataLoader
        The DataLoader for the test or train dataset
    device : str
        The device to run the model on (e.g. 'cuda' or 'cpu' or 'mps')

    Returns
    -------
    tuple
        A tuple containing the Dice similarity coefficients, accuracies, all images, all masks, and all predictions
    """
    model.eval()
    dice_scores = []
    accuracies = []
    all_predictions = []
    all_masks = []
    all_images = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluation (part d)", leave=False)
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)
            predictions = model(images)
            predictions = torch.sigmoid(predictions)

            pred_binary = (predictions > 0.5).float()  # Threshold predictions

            # Store CPU-bound tensors for plotting/analysis
            all_images.extend(images.cpu().numpy())
            all_masks.extend(masks.cpu().numpy())
            all_predictions.extend(pred_binary.cpu().numpy())

            # Calculate metrics for each slice
            for i in range(images.size(0)):
                dice_score = dice_coefficient(pred_binary[i], masks[i])
                dice_scores.append(dice_score.item())

                correct = (pred_binary[i] == masks[i]).float().sum()
                accuracy = correct / (masks[i].shape[1] * masks[i].shape[2])
                accuracies.append(accuracy.item())

    return dice_scores, accuracies, all_images, all_masks, all_predictions


def get_top_worst_med_scores(dice_scores):
    """
    Get the indices of the top 3, worst 3 and median 3 DSC scores.

    Parameters
    ----------
    dice_scores : list
        List of DSC scores.

    Returns
    -------
    best_indices : list
        Indices of the top 3 DSC scores.
    worst_indices : list
        Indices of the worst 3 DSC scores.
    median_indices : list
        Indices of the median 3 DSC scores.
    """
    sorted_indices = np.argsort(dice_scores)
    num_slices = len(dice_scores)

    # Best DSC values (highest scores)
    best_indices = sorted_indices[-3:]

    # Worst DSC values (lowest scores)
    worst_indices = sorted_indices[:3]

    # Median DSC values
    median_indices = sorted_indices[num_slices // 2 - 1 : num_slices // 2 + 2]

    return best_indices, worst_indices, median_indices
