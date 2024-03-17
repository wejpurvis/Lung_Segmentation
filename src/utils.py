"""
Utility functions for Medical Imaging project
"""

import os
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
    numpy.ndarray
        A 3d numpy array of the patient's DICOM files
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
                    dicom_data["PatientName"] = case_id
                    dicom_data["PatientID"] = case_id
                    dicom_data["PatientBirthDate"] = ""
                    try:
                        del dicom_data["PatientBirthTime"]
                    except KeyError:
                        pass
                    # Save anonymized DICOM files
                    dicom_data.save_as(full_path)

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
        A dictionary of DICOMS for each patient
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

    # Check if DICOM and Segmentation pickle files are in data folder
    if os.path.exists(seg_pickle_path) and os.path.exists(dic_pickle_path):
        print("DICOM and segmentation data found, loading data from pickle files")
        with open(seg_pickle_path, "rb") as file:
            segs = pickle.load(file)
        with open(dic_pickle_path, "rb") as file:
            dics = pickle.load(file)
    else:
        print(
            "Compressed DICOM and segmentation data not found, loading data from .npz and DICOM files"
        )

        # Add validation to check image and segmentation subdirectories exist
        if not os.path.exists(os.path.join(data_path, "Segmentations")):
            raise FileNotFoundError("Segmentation data not found")
        if not os.path.exists(os.path.join(data_path, "Images")):
            raise FileNotFoundError("DICOM data not found")

        seg_path = os.path.join(data_path, "Segmentations")
        dic_path = os.path.join(data_path, "Images")

        segs = load_patient_segmentations(seg_path, loading=True)
        dics = load_all_patient_dicoms_para(dic_path, loading=True)

        # Save the data as pickle files
        with open(seg_pickle_path, "wb") as file:
            pickle.dump(segs, file)
        with open(dic_pickle_path, "wb") as file:
            pickle.dump(dics, file)

    print("Data loaded successfully")
    return dics, segs
