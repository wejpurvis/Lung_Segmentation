"""
Plotting functions for medical imaging CW
"""

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, "plots")

plt.rcParams["savefig.dpi"] = 300  # high res images


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


def plot_scores(dice_scores, accuracies, name, save=True):
    """
    Plot the dice scores and accuracies for each slice in the dataset (part d).

    Parameters
    ----------
    dice_scores : list
        List of dice scores for each slice
    accuracies : list
        List of accuracies for each slice
    name : str
        Name of the dataset (to be shown on plot title)
    save : bool, optional
        Whether to save the plot as a .png file (default is True)
    """
    plt.figure(figsize=(20, 6))

    # Add suptitle
    plt.suptitle(f"Model Evaluation Metrics for {name} dataset", fontsize=16)

    # Dice Score Plot
    plt.subplot(1, 2, 1)
    plt.plot(dice_scores, label="Dice Score", color="slategrey")
    plt.xlabel("Slice number")
    plt.ylabel("Dice Score")
    plt.title("Dice Score per Slice")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label="Binary Accuracy", color="dodgerblue")
    plt.xlabel("Slice number")
    plt.ylabel("Accuracy")
    plt.title("Binary Accuracy per Slice")
    plt.legend()

    if save:
        print(f"Saving plot as plots/{name}_scores.png")
        plt_filename = f"{name}_scores.png"
        plt.savefig(os.path.join(plots_dir, plt_filename), bbox_inches="tight")
    return None


def plot_examples(
    indices,
    dice_scores,
    accuracies,
    masks,
    predictions,
    segmentations,
    title,
    save=True,
):
    """
    Plot examples of segmentations, masks, and predictions for a given set of indices.

    Parameters
    ----------
    indices : list
        List of indices to plot
    dice_scores : list
        List of dice scores for each slice
    accuracies : list
        List of accuracies for each slice
    masks : list
        List of masks for each slice (ground truth)
    predictions : list
        List of predicted masks for each slice
    segmentations : list
        List of segmentations for each slice (CT scan)
    title : str
        Title of the plot
    save : bool, optional
        Whether to save the plot as a .png file (default is True)
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, idx in enumerate(indices):
        # Plot segmentation
        axes[i, 0].imshow(segmentations[idx][0], cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Segmentation", fontsize=14)
        # Add DSC score as text
        axes[i, 0].text(
            0.95,
            0.05,
            f"DSC: {dice_scores[idx]:.2f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axes[i, 0].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )
        # Add slice number
        axes[i, 0].text(
            0.05,
            0.05,
            f"Slice: {idx}",
            verticalalignment="top",
            horizontalalignment="left",
            transform=axes[i, 0].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )
        # Add accuracy as text
        axes[i, 0].text(
            0.05,
            0.95,
            f"Acc: {accuracies[idx]*100:.2f}%",
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=axes[i, 0].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )

        # Plot mask
        axes[i, 1].imshow(masks[idx][0], cmap="gray")
        axes[i, 1].set_title("Mask", fontsize=14)
        axes[i, 1].axis("off")
        axes[i, 1].text(
            0.95,
            0.05,
            f"DSC: {dice_scores[idx]:.2f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axes[i, 1].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )
        axes[i, 1].text(
            0.05,
            0.95,
            f"Acc: {accuracies[idx]*100:.2f}%",
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=axes[i, 1].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )

        # Plot prediction
        axes[i, 2].imshow(predictions[idx][0], cmap="gray")
        axes[i, 2].set_title("Prediction", fontsize=14)
        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.95,
            0.05,
            f"DSC: {dice_scores[idx]:.2f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axes[i, 2].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )
        axes[i, 2].text(
            0.05,
            0.95,
            f"Acc: {accuracies[idx]*100:.2f}%",
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=axes[i, 2].transAxes,
            color="white",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="black", edgecolor="black", alpha=0
            ),
        )

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        print(f"Saving plot as plots/{title}.png")
        plt_filename = f"{title}.png"
        plt.savefig(os.path.join(plots_dir, plt_filename), bbox_inches="tight")
    plt.show()
