# Import the PyTorch neural network module
import torch.nn as nn

# Import the building blocks used to construct the U-Net
from .unet_parts import DoubleConv
from .unet_parts import Down
from .unet_parts import Up
from .unet_parts import OutConv


# This class assembles the complete vanilla U-Net architecture.
#
# The network has three stages:
#   1. Encoder - learns increasingly complex image features while reducing
#      the image size.
#   2. Decoder - rebuilds the image to its original size while combining
#      information from the encoder using skip connections.
#   3. Output layer - produces one prediction for every pixel.
#
# This implementation is designed for binary semantic segmentation,
# where each pixel is classified as either lava flow or background.
class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        #
        # The encoder gradually reduces the image size while increasing
        # the number of feature maps.
        #
        # This allows the network to recognise increasingly complex
        # structures in the image.

        # Initial feature extraction
        self.inc = DoubleConv(2, 64)

        # Downsampling stages
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # -------------------------
        # Decoder
        # -------------------------
        #
        # The decoder gradually restores the image to its original size.
        # At each stage it combines information from the encoder using
        # skip connections to recover fine image detail.

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # -------------------------
        # Output layer
        # -------------------------
        #
        # Reduce the final 64 feature maps down to a single output channel.
        # The output contains one value for every pixel in the image.

        self.outc = OutConv(64, 2)

    def forward(self, x):

        # Pass the input image through the encoder.
        #
        # The outputs are stored because they will later be reused by the
        # decoder through the skip connections.

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Pass through the decoder.
        #
        # At each stage the decoder combines the upsampled feature maps
        # with the corresponding encoder feature maps.

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Produce the final output.
        #
        # The output values are called logits.
        # They are not probabilities yet.
        # During training, BCEWithLogitsLoss converts these logits into
        # probabilities internally using a sigmoid function.

        logits = self.outc(x)

        return logits