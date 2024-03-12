"""
Plotting functions for medical imaging CW
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_patients(
    dicoms: dict,
    segmentations: dict,
    overlay: bool = True,
    save: bool = False,
    animate: bool = False,
) -> None:
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
    save : bool, optional
        Whether to save the plot as a .png file (default is False)
    animate : bool, optional
        Whether to animate the images (default is False) NOTE: cannot be used with
        overlay=False

    """
    if (overlay is False) and (animate is True):
        raise ValueError("Cannot animate with overlay=False")

    if animate:
        filename = "CT_Scans_Overlay_Animated"
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        axs = axs.ravel()

        def update(frame: int) -> None:
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

        # interval dictates speed of animation (higher -> slower)
        ani = FuncAnimation(fig, update, frames=np.arange(50, 300, 2), interval=1)
        if save:
            print(f"Saving animation as plots/{filename}.gif")
            ani.save(f"plots/{filename}.gif", writer="pillow", fps=20)
        plt.show()
        return ani

    else:
        # Modify this to change position along coronal plane
        slide = 250
        if overlay:
            filename = "CT_Scans_Overlay_Static"
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            axs = axs.ravel()

            for i, case_id in enumerate(dicoms.keys()):
                case_dicom = dicoms[case_id]
                case_segmentation = segmentations[case_id]
                axs[i].imshow(case_dicom[:, slide, :], cmap="gray", aspect="auto")
                axs[i].imshow(
                    case_segmentation[:, slide, :],
                    cmap="viridis",
                    alpha=0.3,
                    aspect="auto",
                )
                axs[i].axis("off")  # Turn the axis off
                axs[i].set_title(case_id)  # Add title
                axs[i].text(
                    0.05,
                    0.95,
                    f"Slide {slide}",
                    color="white",
                    transform=axs[i].transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )

        else:
            filename = "CT_Scans_Static"
            fig, axs = plt.subplots(12, 2, figsize=(6, 36))
            for i, case_id in enumerate(dicoms.keys()):
                case_dicom = dicoms[case_id]
                case_segmentation = segmentations[case_id]
                axs[i, 0].imshow(case_dicom[:, slide, :], cmap="gray", aspect="auto")
                axs[i, 1].imshow(
                    case_segmentation[:, slide, :], cmap="viridis", aspect="auto"
                )
                axs[i, 0].axis("off")  # axis off for dicom
                axs[i, 1].axis("off")  # axis off for segmentation
                axs[i, 0].text(
                    0.05,
                    0.95,
                    f"Slide {slide}",
                    color="white",
                    transform=axs[i, 0].transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )
                axs[i, 1].text(
                    0.05,
                    0.95,
                    f"Slide {slide}",
                    color="white",
                    transform=axs[i, 1].transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                )
                axs[i, 0].set_title(case_id + " DICOM")  # title for dicom
                axs[i, 1].set_title(case_id + " Segmentation")  # title for segmentation

    plt.tight_layout()
    if save:
        print(f"Saving plot as plots/{filename}.png")
        plt.savefig(f"plots/{filename}.png")
    plt.show()
    return None
