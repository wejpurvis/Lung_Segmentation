<!-- omit in toc -->
# Medical Imaging Coursework

**Author**: William Purvis

<!-- omit in toc -->
## Table of Contents

- [Project description](#project-description)
- [How to install and run the program](#how-to-install-and-run-the-program)
  - [Installing locally](#installing-locally)
  - [Running the program](#running-the-program)
    - [Module 1: Handling DICOM data](#module-1-handling-dicom-data)
    - [Module 2b: UNet-based segmentation](#module-2b-unet-based-segmentation)
  - [Building documentation](#building-documentation)
- [License](#license)

![Segmentations of lung cancers](plots/CT_Scans_Overlay_Animated.gif)

## Project description

This package is split into two parts: the first module, `load_dicom.py`, converts a DICOM dataset of 12 cases from the LCTSC Lung CT Segmentation Challenge into 3D NumPy arrays to be used for further analysis.

The second part of this project trains a U-Net model to segment lungs in the CT images provided and calculates various metrics such as binary accuracy, training loss, and Dice similarity coefficient between predicted lung segments and actual segmentations to investigate the efficacy of the trained model.

## How to install and run the program

To clone the repository, run the following command:

```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A2_MED_Assessment/wp289.git
```

Next, navigate to the `data` subdirectory and download the DICOM and segmentation datasets from the [Course GitHub](https://github.com/loressa/DataScience_MPhill_practicals/tree/master/Dataset). If these datasets are not downloaded, a `FileNotFoundError` is raised when running the project.

### Installing locally

As this project requires external datasets to be downloaded, to avoid volume mounting or downloading these files at build-time, containerisation was carried out at the Conda level instead of using Docker.

To run this package locally, create a Conda environment from the provided `environment.yml` file:

```bash
# create env
conda env create -f environment.yml
# activate env
conda activate medical_imaging_wp289
```

### Running the program

#### Module 1: Handling DICOM data

To convert the DICOM dataset into NumPy arrays per case (Module 1), run the following command from the root directory:

```bash
python src/load_dicom.py
```

If the DICOM dataset has been installed correctly, the script will generate a `dicoms.pkl` file in the `data` subdirectory, containing the generated 3D NumPy arrays per case.

#### Module 2b: UNet-based segmentation

To run the UNet model, the following command can be used from the root directory:

```bash
python src/main.py
```

This script will first verify whether the DICOM and segmentation datasets have been downloaded, and if so, convert them into NumPy arrays as in Module 1. Next, the data is pre-processed and loaded into a training module which trains a U-Net model over 10 epochs using a **custom loss function**.

To run the U-Net model with a standard binary cross entropy loss, run the following command instead:

```bash
python src/main.py --default-loss
```

Once the model has been trained, it's weights and parameters are saved as a `.pth` file to the `models/` subdirectory. Additionally, the losses and accuracies per epoch are saved to the same subdirectory.

Finally, the script generates various plots used in the report and saves them to the `plots/` subdirectory.

### Building documentation

Documentation for this project can be built using Sphinx by running the following commands from the root directory:

```bash
cd docs

make html
```

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt) file for details.