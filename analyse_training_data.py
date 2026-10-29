# ==========================================================
# TRAINING DATA ANALYSIS
# ==========================================================
#
# This script analyses the tiled segmentation masks used to
# train the lava-only U-Net model.
#
# The script reports:
#   - The total number of training masks.
#   - The number and percentage of pixels belonging to each class.
#   - How many tiles contain each class.
#   - The amount of lava coverage within lava-containing tiles.
#   - Suggested class weights using median-frequency balancing.
#
# This script does not modify any masks or training data.
#
# The expected mask pixel values are:
#   0   = Non-lava
#   255 = Lava
#
# Run this script after changing or regenerating the training
# dataset so that the class distribution and suggested class
# weights can be checked before training the model.


# ==========================================================
# IMPORTS
# ==========================================================

# Path is used to define folder locations and find mask files.
from pathlib import Path

# NumPy is used for pixel counting and statistical calculations.
import numpy as np

# PIL is used to open the PNG mask images.
from PIL import Image


# ==========================================================
# SETTINGS
# ==========================================================

# Folder containing the tiled training masks.
MASK_FOLDER = Path("data/training/masks")

# Class definitions.
#
# Each mask is a single-channel image in which the pixel value
# identifies the segmentation class.
CLASS_NAMES = {
    0: "Non-lava",
    255: "Lava",
}


# ==========================================================
# FIND TRAINING MASKS
# ==========================================================

# Find all PNG files in the training mask folder.
# Sorting ensures that files are processed in a consistent order.
mask_files = sorted(MASK_FOLDER.glob("*.png"))

# Stop immediately if no masks are found. This prevents the
# calculations below from running on an empty dataset.
if not mask_files:
    raise FileNotFoundError(
        f"No PNG masks found in {MASK_FOLDER}"
    )

# Report how many training masks will be analysed.
print(f"Training masks found: {len(mask_files)}")


# ==========================================================
# COUNT PIXELS FOR EACH CLASS
# ==========================================================

# Initialise a running pixel count for each mask class.
class_pixel_counts = {
    0: 0,
    255: 0,
}

# Running count of all pixels across all training masks.
total_pixels = 0

# Process each training mask individually.
for mask_path in mask_files:

    # Open the mask and convert it to a NumPy array.
    mask = np.array(Image.open(mask_path))

    # Count all pixels in this mask.
    total_pixels += mask.size

    # Count the pixels belonging to each class and add them
    # to the running total for that class.
    for class_idx in CLASS_NAMES:
        class_pixel_counts[class_idx] += np.sum(mask == class_idx)


# ==========================================================
# PRINT CLASS DISTRIBUTION
# ==========================================================

# Report the total number of pixels across the training dataset.
print()
print(f"Total training pixels: {total_pixels:,}")
print()

# Calculate and report the pixel count and percentage occupied
# by each segmentation class.
for class_idx, class_name in CLASS_NAMES.items():

    pixel_count = class_pixel_counts[class_idx]

    percentage = (
        pixel_count / total_pixels
    ) * 100

    print(
        f"{class_name}: "
        f"{pixel_count:,} pixels "
        f"({percentage:.2f}%)"
    )


# ==========================================================
# COUNT TILES CONTAINING EACH CLASS
# ==========================================================

tiles_with_background = 0
tiles_with_lava = 0
tiles_with_no_foreground = 0

# Check the class content of every training mask.
for mask_path in mask_files:

    # Open the mask and convert it to a NumPy array.
    mask = np.array(Image.open(mask_path))

    # Determine whether each class is present.
    has_background = np.any(mask == 0)
    has_lava = np.any(mask == 255)

    if has_background:
        tiles_with_background += 1

    if has_lava:
        tiles_with_lava += 1

    # Tiles without lava contain only non-lava pixels.
    if not has_lava:
        tiles_with_no_foreground += 1
# ==========================================================
# PRINT TILE DISTRIBUTION
# ==========================================================

print()
print("Tile distribution:")
print()

print(
    f"Tiles containing non-lava: "
    f"{tiles_with_background} "
    f"({tiles_with_background / len(mask_files) * 100:.2f}%)"
)

print(
    f"Tiles containing lava: "
    f"{tiles_with_lava} "
    f"({tiles_with_lava / len(mask_files) * 100:.2f}%)"
)

print(
    f"Tiles containing no lava: "
    f"{tiles_with_no_foreground} "
    f"({tiles_with_no_foreground / len(mask_files) * 100:.2f}%)"
)

# ==========================================================
# LAVA COVERAGE WITHIN LAVA-CONTAINING TILES
# ==========================================================

lava_coverages = []

for mask_path in mask_files:

    # Open the mask and convert it to a NumPy array.
    mask = np.array(Image.open(mask_path))

    # Count lava pixels.
    lava_pixels = np.sum(mask == 255)

    # Only include tiles containing lava.
    if lava_pixels > 0:

        lava_percentage = (lava_pixels / mask.size) * 100
        lava_coverages.append(lava_percentage)

lava_coverages = np.array(lava_coverages)


# ==========================================================
# PRINT LAVA COVERAGE STATISTICS
# ==========================================================

print()
print("Lava coverage in lava-containing tiles:")
print()

print(f"Number of lava-containing tiles: {len(lava_coverages)}")
print(f"Minimum lava coverage: {lava_coverages.min():.2f}%")
print(f"25th percentile:       {np.percentile(lava_coverages, 25):.2f}%")
print(f"Median lava coverage:  {np.median(lava_coverages):.2f}%")
print(f"Mean lava coverage:    {lava_coverages.mean():.2f}%")
print(f"75th percentile:       {np.percentile(lava_coverages, 75):.2f}%")
print(f"Maximum lava coverage: {lava_coverages.max():.2f}%")


# ==========================================================
# CALCULATE STARTING CLASS WEIGHTS
# ==========================================================

# Calculate the fraction of all training pixels belonging to
# each segmentation class.
class_frequencies = {
    class_value: class_pixel_counts[class_value] / total_pixels
    for class_value in CLASS_NAMES
}

# Calculate class weights using median-frequency balancing.
#
# For each class:
#
# weight = median class frequency / class frequency
#
# Classes that occupy a smaller proportion of the training
# dataset therefore receive a larger suggested weight.
median_frequency = np.median(list(class_frequencies.values()))

class_weights = {
    class_value: median_frequency / frequency
    for class_value, frequency in class_frequencies.items()
}


# ==========================================================
# PRINT SUGGESTED CLASS WEIGHTS
# ==========================================================

# Display the calculated weight for each class.
print()
print("Suggested class weights using median-frequency balancing:")
print()

for class_value, class_name in CLASS_NAMES.items():
    print(
        f"{class_name}: "
        f"{class_weights[class_value]:.4f}"
    )

# Print the weights in the same class order used by the model:
# Background, Faults, Lava.
#
# This line can be copied into the training script as the
# starting CLASS_WEIGHTS values if required.
print()
print(
    "CLASS_WEIGHTS = ["
    f"{class_weights[0]:.4f}, "
    f"{class_weights[255]:.4f}"
    "]"
)