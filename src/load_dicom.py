"""
This module is the entrypoint for Module 1 of the Medical Imaging coursework: Handling Dicom data.

Running python src/load_dicom.py from the root directory will generate a dictionary of numpy arrays per case and save them to the `data/` directory as a pickle file. Each numpy array represents the 3D DICOM data for a particular patient and has shape [x, y, z] where x is the number of slices (axial plane), y is the coronal plane, and z is the sagittal plane.
"""

import os
import pickle
from utils import load_all_patient_dicoms_para, get_dicom_args


def generate_DICOM_dict(load: bool, anonymize: bool) -> dict:
    """
    Generates a dictionary of patient IDs to their 3D DICOM data numpy arrays.

    Parameters
    ----------
    load : bool
        Whether to display progress of loading DICOM files in terminal
    anonymize : bool
        Whether to anonymize the patient's data when loading the DICOM files

    Returns
    -------
    dict
        Dictionary of patient IDs to their 3D DICOM data numpy arrays
    """
    # Path to the data directory
    data_path = os.path.join(os.getcwd(), "data")

    dics = None

    # Path for pickle file
    dic_pickle_path = os.path.join(data_path, "dicoms.pkl")

    # Check if DICOM pickle file already exists
    if os.path.exists(dic_pickle_path):
        print("Processed DICOM data found, loading from pickle file...")
        with open(dic_pickle_path, "rb") as f:
            dics = pickle.load(f)
    else:
        # Load DICOM data
        print("Compressed DICOM data not found, loading data from DICOM files.")
        # Check DICOM files exist
        if not os.path.exists(os.path.join(data_path, "Images")):
            raise FileNotFoundError(
                "\n\nDICOM data not found. Please download the 'Images' subdirectory and place it in the 'data' directory.\n"
            )

        dic_path = os.path.join(data_path, "Images")
        dics = load_all_patient_dicoms_para(dic_path, loading=load, anonymize=anonymize)

        # Save the data as a pickle file
        with open(dic_pickle_path, "wb") as f:
            pickle.dump(dics, f)
            print("Data saved to 'data/dicoms.pkl'")

    print("DICOM data loaded successfully.")
    return dics


if __name__ == "__main__":
    # Parse command line arguments
    args = get_dicom_args()
    anonymize = args.anonymize
    load = args.show_progress

    # Generate the DICOM data dictionary
    dicom_dict = generate_DICOM_dict(load, anonymize)
