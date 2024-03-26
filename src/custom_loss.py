r"""
This module contains custom loss functions combinig cross entropy loss and soft dice loss.

The dice similarity coefficient (DSC) is a spatial overlap index that measures the similarity between two binary images. This measure ranges from 0 to 1, where 0 indicates no spatial overlap and 1 indicates perfect spatial overlap. The DSC is defined as:

.. math::
    Dice = \frac{2\left | A\cap B \right |}{\left | A \right | + \left |  B \right |}

where :math:`\left | A\cap B \right |` represents the common elements between sets A and B, and :math:`\left | A \right |` and :math:`\left | B \right |` represent the number of elements in sets A and B, respectively.

:math:`\left | A\cap B \right |` can be approximated by the sum of the element-wise product of the two binary images (prediction and target mask).
:math:`\left | A \right |` and :math:`\left | B \right |` can either be quantified by the sum of the elements in the binary images or by the square of the sum of the elements in the binary images.

In order to formulate a loss function which can be minimised, the DSC can be subtracted from 1. This is known as the **soft dice loss**:

.. math::
    SoftDiceLoss = 1 - \frac{2 (\sum_{pixels} y_{true}y_{pred})+\epsilon}{\sum_{pixels} y_{true}^{2}\sum_{pixels} y_{pred}^{2} + \epsilon}

where :math: `\epsilon` is a small constant to avoid division by zero.

The **cross entropy loss** is a standard loss function used for binary classification problems. It is defined as:

.. math::
    CrossEntropyLoss = -\sum_{pixels} y_{true}log(y_{pred}) + (1-y_{true})log(1-y_{pred})

where :math:`y_{true}` is the true label and :math:`y_{pred}` is the predicted label.
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

    def forward(self, outputs, targets):
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
