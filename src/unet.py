"""
UNet model for image segmentation.
"""

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """
    Initialises a SimpleUNet model

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels

    Attributes
    ----------
    conv1 : nn.Sequential
        First convolutional block
    maxpool1 : nn.Sequential
        First maxpooling block
    conv2 : nn.Sequential
        Second convolutional block
    maxpool2 : nn.Sequential
        Second maxpooling block
    conv3 : nn.Sequential
        Third convolutional block
    maxpool3 : nn.Sequential
        Third maxpooling block
    middle : nn.Sequential
        Middle convolutional block
    upsample3 : nn.ConvTranspose2d
        Third upsampling block
    upconv3 : nn.Sequential
        Third upconvolutional block
    upsample2 : nn.ConvTranspose2d
        Second upsampling block
    upconv2 : nn.Sequential
        Second upconvolutional block
    upsample1 : nn.ConvTranspose2d
        First upsampling block
    upconv1 : nn.Sequential
        First upconvolutional block
    final : nn.Conv2d
        Final layer

    Methods
    -------
    conv_block(in_channels, out_channels, kernel_size, stride, padding)
        Returns a convolutional block
    maxpool_block(kernel_size, stride, padding)
        Returns a maxpooling block
    transposed_block(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        Returns a transposed convolutional block
    final_layer(in_channels, out_channels, kernel_size, stride, padding)
        Returns the final layer
    forward(x)
        Forward pass through the model
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 16, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(16, 32, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 0)
        self.conv3 = self.conv_block(32, 64, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 0)

        self.middle = self.conv_block(64, 128, 3, 1, 1)

        self.upsample3 = self.transposed_block(128, 64, 3, 2, 1, 1)
        self.upconv3 = self.conv_block(128, 64, 3, 1, 1)
        self.upsample2 = self.transposed_block(64, 32, 3, 2, 1, 1)
        self.upconv2 = self.conv_block(64, 32, 3, 1, 1)

        self.upsample1 = self.transposed_block(32, 16, 3, 2, 1, 1)
        self.upconv1 = self.conv_block(32, 16, 3, 1, 1)

        self.final = self.final_layer(16, 1, 1, 1, 0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates a convolutional block consisting of two convolutional layers followed by batch normalization and ReLU activation.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Size of the kernel
        stride : int
            Stride of the kernel
        padding : int
            Padding of the kernel


        Returns
        -------
            nn.Sequential
                Sequential container of the convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def maxpool_block(self, kernel_size, stride, padding):
        """
        Creates a maxpool block with the specified kernel size, stride, and padding.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel
        stride : int
            Stride of the kernel
        padding : int
            Padding of the kernel

        Returns
        -------
        nn.Sequential
            A sequential module consisting of a maxpooling layer
            followed by a dropout layer.
        """
        maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout2d(0.5),
        )
        return maxpool

    def transposed_block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        """
        Creates a transposed block for the UNet model.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Size of the kernel
        stride : int
            Stride of the kernel
        padding : int
            Padding of the kernel
        output_padding : int
            Output padding of the kernel

        Returns
        -------
        nn.ConvTranspose2d
            A transposed convolutional layer.
        """
        transposed = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Creates the final layer of the UNet model.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Size of the kernel
        stride : int
            Stride of the kernel
        padding : int
            Padding of the kernel

        Returns
        -------
            nn.Conv2d: The final layer of the UNet model.
        """
        final = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # downsampling part
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # middle part
        middle = self.middle(maxpool3)
        # upsampling part
        upsample3 = self.upsample3(middle)
        upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
        upsample2 = self.upsample2(upconv3)
        upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))
        final_layer = self.final(upconv1)
        return final_layer
