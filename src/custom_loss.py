r"""
This module contains the custom loss function which combines cross entropy loss and soft dice loss.

The dice similarity coefficient (DSC) is a spatial overlap index that measures the similarity between two binary images. This measure ranges from 0 to 1, where 0 indicates no spatial overlap and 1 indicates perfect spatial overlap. The DSC is defined as:

.. math::
    \textrm{DSC} = \frac{2\left | A\cap B \right |}{\left | A \right | + \left |  B \right |}

where :math:`\left | A\cap B \right |` represents the common elements between sets A and B, and :math:`\left | A \right |` and :math:`\left | B \right |` represent the number of elements in sets A and B, respectively.

:math:`\left | A\cap B \right |` can be approximated by the sum of the element-wise product of the two binary images (prediction and target mask).
:math:`\left | A \right |` and :math:`\left | B \right |` can either be quantified by the sum of the elements in the binary images or by the square of the sum of the elements in the binary images.

In order to formulate a loss function which can be minimised, the DSC can be subtracted from 1. This is known as the **soft dice loss**:

.. math::
    \textrm{SoftDiceLoss} = 1 - \frac{2 (\sum_{i=1}^{N} y_{i}\hat{y_{i}})+\epsilon}{\sum_{i=1}^{N} y_{i}^{2}\sum_{i}^{N} \hat{y_{i}}^{2} + \epsilon}

where :math:`\epsilon` is a small constant to avoid division by zero.

The **cross entropy loss** is a standard loss function used for binary classification problems. It is defined as:

.. math::
    \textrm{CrossEntropyLoss} = -\frac{1}{N}\sum_{i=1}^{N} y_{i}\log(\hat{y_{i}}) + (1-y_{i})\log(1-\hat{y_{i}})

where :math:`y_{i}` is the true label and :math:`\hat{y_{i}}` is the predicted label.
"""

import torch
import torch.nn.functional as F


class CombinedLoss(torch.nn.Module):
    """
    Custom loss function which combines cross entropy loss and soft dice loss.

    Parameters
    ----------
    alpha : float, optional
        The weighting factor for the cross entropy loss (default is 0.5)
    epsilon : float, optional
        A small constant to avoid division by zero (default is 1e-6)
    """

    def __init__(self, alpha=0.5, epsilon=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.epsilon = epsilon
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the combined loss function.

        Parameters
        ----------
        outputs : torch.Tensor
            The output of the model
        targets : torch.Tensor
            The target mask

        Returns
        -------
        torch.Tensor
            The combined loss value
        """
        bce_loss = self.bce_loss(outputs, targets)

        # Soft dice loss (apply sigmoid to the output)
        outputs_sig = torch.sigmoid(outputs)
        # Compute Soft Dice Loss for each item in the batch
        num = 2 * torch.sum(outputs_sig * targets, dim=[1, 2, 3]) + self.epsilon
        den = (
            torch.sum(outputs_sig**2, dim=[1, 2, 3])
            + torch.sum(targets**2, dim=[1, 2, 3])
            + self.epsilon
        )
        soft_dice_loss = 1 - (num / den)

        # Average the Soft Dice Loss across the batch
        soft_dice_loss = torch.mean(soft_dice_loss)

        combined_loss = (self.alpha * bce_loss) + (self.beta * soft_dice_loss)
        return combined_loss
