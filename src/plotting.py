"""
Plotting functions for medical imaging CW
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_patients(dicoms, segmentations, overlay=True, animate=False):
    """
    Plot CT scans for patients with or without segmentations overlaid

    Parameters
    ----------
    dicoms : dict
        A dictionary of DICOMS for each patient (savd as 3D numpy arrays)
    segmentations : dict
        A dictionary of segmentations for each patient
    overlay : bool, optional
        Whether to overlay segmentations on DICOM images (default is True)
    animate : bool, optional
        Whether to animate the images (default is False)

    """
    if (overlay is False) and (animate is True):
        raise ValueError("Cannot animate with overlay=False")

    if animate:
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        axs = axs.ravel()

        def update(frame):
            for ax in axs:
                ax.clear()  # Clear the previous image
            for i, case_id in enumerate(dicoms.keys()):
                case_dicom = dicoms[case_id]
                case_segmentation = segmentations[case_id]
                if overlay:
                    axs[i].imshow(case_dicom[:, frame, :], cmap="gray", aspect="auto")
                    axs[i].imshow(
                        case_segmentation[:, frame, :],
                        cmap="viridis",
                        alpha=0.3,
                        aspect="auto",
                    )
                else:
                    # Cannot animate without overlay
                    pass
                axs[i].axis("off")
                axs[i].set_title(case_id)
                axs[i].text(
                    0.05,
                    0.95,
                    f"Slide {frame + 1}",
                    color="white",
                    transform=axs[i].transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )

            plt.tight_layout()

        # interval dictates speed of animation
        ani = FuncAnimation(fig, update, frames=np.arange(50, 513), interval=300)
        plt.show()
        return ani

    else:
        if overlay:
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            axs = axs.ravel()

            for i, case_id in enumerate(dicoms.keys()):
                case_dicom = dicoms[case_id]
                case_segmentation = segmentations[case_id]
                axs[i].imshow(case_dicom[:, 250, :], cmap="gray", aspect="auto")
                axs[i].imshow(
                    case_segmentation[:, 250, :],
                    cmap="viridis",
                    alpha=0.3,
                    aspect="auto",
                )
                axs[i].axis("off")  # Turn the axis off
                axs[i].set_title(case_id)  # Add title

        else:
            fig, axs = plt.subplots(12, 2, figsize=(6, 36))
            for i, case_id in enumerate(dicoms.keys()):
                case_dicom = dicoms[case_id]
                case_segmentation = segmentations[case_id]
                axs[i, 0].imshow(case_dicom[:, 250, :], cmap="gray", aspect="auto")
                axs[i, 1].imshow(
                    case_segmentation[:, 250, :], cmap="viridis", aspect="auto"
                )
                axs[i, 0].axis("off")  # Turn the axis off for dicom
                axs[i, 1].axis("off")  # Turn the axis off for segmentation
                axs[i, 0].set_title(case_id + " DICOM")  # Add title for dicom
                axs[i, 1].set_title(
                    case_id + " Segmentation"
                )  # Add title for segmentation

    plt.tight_layout()  # Make layout tight
    plt.show()
    return None
