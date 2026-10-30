
"""
==========================================================
U-NET DATASET LOADING
==========================================================

This file defines the custom PyTorch Dataset used to load
the tiled SAR images, texture maps and segmentation masks.

Each dataset sample contains:

  - One greyscale SAR image tile.
  - Its corresponding float32 texture tile.
  - Its corresponding segmentation mask.
  - The original SAR tile filename.

INPUT CHANNELS
--------------

Channel 0 = SAR intensity
    Stored as uint8 PNG.
    Converted to float32 and scaled from 0-255 to 0-1.

Channel 1 = SAR texture
    Stored as float32 TIFF.
    Original texture ratios are preserved.
    No additional normalisation is applied.

The two channels are combined into a tensor with shape:

    [2, height, width]

SEGMENTATION MASKS
------------------

Masks stored on disk contain:

    0   = Everything else
    255 = Lava

These are converted into one binary channel:

    Channel 0 = Lava

Non-lava pixels have a value of zero.

DATA AUGMENTATION
-----------------

When augment=True:

- Random rotations of 0, 90, 180 or 270 degrees.
- Random horizontal flip with probability 0.5.
- Random vertical flip with probability 0.5.
- Multiplicative SAR brightness augmentation with
  probability 0.5 and multiplier range 0.7-1.3.

Geometric transformations are applied identically to
both input channels and the segmentation mask.

Brightness augmentation is applied only to SAR channel 0.

==========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

from email.mime import image
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import torch
import rasterio


# ==========================================================
# DATASET CLASS
# ==========================================================

class LavaFlowDataset(Dataset):

    # ======================================================
    # INITIALISE DATASET
    # ======================================================

    def __init__(
        self,
        images_dir,
        masks_dir,
        texture_dir,
        augment=False,
        brightness_augment=False
    ):

        # Store the locations of the image, mask and
        # texture folders.

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.texture_dir = Path(texture_dir)

        # Store whether augmentation should be applied.
        #
        # Training can use augment=True.
        # Validation and testing can use augment=False.

        self.augment = augment
        self.brightness_augment = brightness_augment
        # Find every PNG image in the image folder.
        #
        # Sorting preserves the original dataset ordering.

        self.image_files = sorted(
            self.images_dir.glob("*.png")
        )

        # Convert SAR images into PyTorch tensors.
        #
        # ToTensor() scales uint8 pixel values
        # from 0-255 to 0-1.

        self.image_transform = transforms.ToTensor()


    # ======================================================
    # RETURN DATASET LENGTH
    # ======================================================

    def __len__(self):

        return len(self.image_files)


    # ======================================================
    # LOAD ONE DATASET SAMPLE
    # ======================================================

    def __getitem__(self, idx):

        # --------------------------------------------------
        # GET MATCHING FILE PATHS
        # --------------------------------------------------

        image_path = self.image_files[idx]

        # Mask tiles use the same filename as SAR tiles.

        mask_path = (
            self.masks_dir / image_path.name
        )

        # Texture tiles use the same filename stem
        # but have a .tif extension.

        texture_path = (
            self.texture_dir
            / f"{image_path.stem}.tif"
        )


        # ==================================================
        # LOAD SAR IMAGE
        # ==================================================

        # Open the SAR image and convert it to greyscale.

        image = Image.open(image_path).convert("L")

        # Convert to a float32 PyTorch tensor.
        #
        # Pixel values are scaled from 0-255 to 0-1.
        #
        # Resulting shape:
        # [1, height, width]

        image = self.image_transform(image)


        # ==================================================
        # LOAD TEXTURE MAP
        # ==================================================

        # Open the corresponding float32 TIFF.

        with rasterio.open(texture_path) as src:

            texture = src.read(1)

        # Convert the texture array to float32.
        #
        # Original texture ratios are preserved.
        # No normalisation or scaling is applied.

        texture = np.asarray(
            texture,
            dtype=np.float32
        )

        # Convert to a PyTorch tensor.
        #
        # Add a channel dimension.
        #
        # Resulting shape:
        # [1, height, width]

        texture = torch.from_numpy(
            texture
        ).unsqueeze(0)


        # ==================================================
        # COMBINE SAR AND TEXTURE CHANNELS
        # ==================================================

        # Combine the SAR and texture tensors.
        #
        # Channel 0 = SAR intensity
        # Channel 1 = Texture
        #
        # Resulting shape:
        # [2, height, width]

        image = torch.cat(
            [image, texture],
            dim=0
        )


        # ==================================================
        # LOAD SEGMENTATION MASK
        # ==================================================

        # Open the corresponding segmentation mask as
        # a single-channel greyscale image.

        mask = Image.open(mask_path).convert("L")

        # Convert the mask into a NumPy array while
        # preserving its integer pixel values.

        mask = np.array(
            mask,
            dtype=np.uint8
        )


        # ==================================================
        # REMAP MASK VALUES TO BINARY CHANNELS
        # ==================================================

        # Masks stored on disk contain:
        #
        # 0   = Everything else
        # 255 = Lava
        #
        # Create one binary target.
        #
        # Channel 0 = Lava
        #
        # All non-lava pixels have a value of zero.

        lava_mask = (
            mask == 255
        ).astype(np.float32)

        # Add a channel dimension to produce:
        #
        # [1, height, width]

        binary_masks = np.expand_dims(
            lava_mask,
            axis=0
        )

        # Convert to a floating-point PyTorch tensor.

        mask = torch.from_numpy(
            binary_masks
        )


        # ==================================================
        # DATA AUGMENTATION
        # ==================================================

        # Apply augmentation during the augmented
        # training pass only.
        #
        # Both SAR and texture are contained in the
        # same image tensor.
        #
        # Geometric transformations therefore affect
        # both channels identically.

        if self.augment:


            # --------------------------------------------------
            # RANDOM ROTATION
            # --------------------------------------------------

            # Randomly choose between:
            #
            # 0 = No rotation
            # 1 = 90 degrees
            # 2 = 180 degrees
            # 3 = 270 degrees

            k = torch.randint(
                0, 4, (1,)
            ).item()

            # Rotate both input channels.

            image = torch.rot90(
                image,
                k,
                dims=(1, 2)
            )

            # Apply the identical rotation to the mask.

            mask = torch.rot90(
                mask,
                k,
                dims=(1, 2)
            )


            # --------------------------------------------------
            # RANDOM HORIZONTAL FLIP
            # --------------------------------------------------

            # Randomly flip horizontally with
            # a 50% probability.

            if torch.rand(1).item() < 0.5:

                # Flip both input channels.

                image = torch.flip(
                    image,
                    dims=(2,)
                )

                # Apply the same horizontal flip to the mask.

                mask = torch.flip(
                    mask,
                    dims=(2,)
                )


            # --------------------------------------------------
            # RANDOM VERTICAL FLIP
            # --------------------------------------------------

            # Randomly flip vertically with
            # a 50% probability.

            if torch.rand(1).item() < 0.5:

                # Flip both input channels.

                image = torch.flip(
                    image,
                    dims=(1,)
                )

                # Apply the same vertical flip to the mask.

                mask = torch.flip(
                    mask,
                    dims=(1,)
                )


            # --------------------------------------------------
            # MULTIPLICATIVE SAR BRIGHTNESS AUGMENTATION
            # --------------------------------------------------

            # Apply brightness augmentation with a
            # probability of 0.5.
            #
            # Only SAR channel 0 is modified.
            #
            # Texture channel 1 remains unchanged.
            #
            # The multiplier is randomly sampled from
            # a uniform distribution between 0.7 and 1.3.

        if self.brightness_augment:

            if torch.rand(1).item() < 0.5:

                brightness_multiplier = (
                    0.7 + 0.6 * torch.rand(1).item()
                )

                # Multiply only the SAR channel.
                #
                # Clamp the result to the existing
                # SAR intensity range of 0-1.

                image[0] = torch.clamp(
                    image[0] * brightness_multiplier,
                    min=0.0,
                    max=1.0
                )


        # ==================================================
        # RETURN SAMPLE
        # ==================================================

        # Return:
        #
        # image:
        #     [1, height, width]
        #
        # mask:
        #     [1, height, width]
        #
        # filename:
        #     Original SAR tile filename

        return image, mask, image_path.name
