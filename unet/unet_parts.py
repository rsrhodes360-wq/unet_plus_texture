# Import PyTorch
import torch

# Import the neural network module
import torch.nn as nn


# This file contains the individual building blocks that make up the U-Net.
# The complete network is assembled from these blocks in unet_model.py.


class DoubleConv(nn.Module):

    # This block applies two convolution layers in succession.
    # Each convolution is followed by a ReLU activation function.
    # Most of the feature extraction in the U-Net happens inside these blocks.

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # nn.Sequential applies each layer one after another.
        self.double_conv = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            # ReLU introduces non-linearity so the network can learn
            # more complex patterns than a simple linear filter.
            nn.ReLU(inplace=True),

            # Second convolution
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            # Second ReLU activation
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # Pass the input through the two convolution layers.
        return self.double_conv(x)


class Down(nn.Module):

    # This block reduces the image size while increasing the number
    # of feature maps.
    #
    # Reducing the image size allows the network to learn larger-scale
    # patterns while keeping the computation manageable.

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(

            # Reduce the image height and width by a factor of two.
            nn.MaxPool2d(2),

            # Extract new features from the smaller image.
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        # Pass the input through the downsampling block.
        return self.maxpool_conv(x)


class OutConv(nn.Module):

    # This is the final layer of the U-Net.
    #
    # A 1 × 1 convolution combines the learned features into the
    # required number of output channels.
    #
    # For binary segmentation we only need one output channel,
    # representing the predicted lava flow probability.

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Produce the final output of the network.
        return self.conv(x)


class Up(nn.Module):

    # This block increases the image size again during the decoder.
    #
    # It combines information from:
    #   - the decoder (large-scale features)
    #   - the encoder (fine spatial detail)
    #
    # This combination is one of the defining features of the U-Net.

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Increase the image height and width by a factor of two.
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        # Refine the combined feature maps.
        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):

        # Upsample the decoder feature maps.
        x1 = self.up(x1)

        # Join the encoder and decoder feature maps together.
        #
        # This is called a skip connection.
        # It allows the decoder to recover fine image detail that
        # would otherwise be lost during downsampling.
        x = torch.cat([x2, x1], dim=1)

        # Refine the combined feature maps.
        return self.conv(x)