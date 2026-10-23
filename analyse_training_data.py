# ==========================================================
# TRAINING DATA ANALYSIS
# ==========================================================
#
# This script analyses the tiled segmentation masks used to
# train the three-class U-Net model.
#
# The script reports:
#   - The total number of training masks.
#   - The number and percentage of pixels belonging to each class.
#   - How many tiles contain each class.
#   - The amount of fault coverage within fault-containing tiles.
#   - Suggested class weights using median-frequency balancing.
#
# This script does not modify any masks or training data.
#
# The expected mask pixel values are:
#   0   = Background
#   128 = Faults
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
    0: "Background",
    128: "Faults",
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
    128: 0,
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

# Initialise counters describing which classes occur within
# individual training tiles.
tiles_with_background = 0
tiles_with_faults = 0
tiles_with_lava = 0
tiles_with_faults_and_lava = 0
tiles_with_no_foreground = 0

# Check the class content of every training mask.
for mask_path in mask_files:

    # Open the mask and convert it to a NumPy array.
    mask = np.array(Image.open(mask_path))

    # Determine whether at least one pixel belonging to each
    # class is present in the current tile.
    has_background = np.any(mask == 0)
    has_faults = np.any(mask == 128)
    has_lava = np.any(mask == 255)

    # Count tiles containing background.
    if has_background:
        tiles_with_background += 1

    # Count tiles containing faults.
    if has_faults:
        tiles_with_faults += 1

    # Count tiles containing lava.
    if has_lava:
        tiles_with_lava += 1

    # Count tiles in which both foreground classes occur.
    if has_faults and has_lava:
        tiles_with_faults_and_lava += 1

    # A tile with neither faults nor lava contains no foreground
    # classes and therefore consists entirely of background.
    if not has_faults and not has_lava:
        tiles_with_no_foreground += 1


# ==========================================================
# PRINT TILE DISTRIBUTION
# ==========================================================

# Report both the number and percentage of training tiles
# containing each class or class combination.
print()
print("Tile distribution:")
print()

print(
    f"Tiles containing background: "
    f"{tiles_with_background} "
    f"({tiles_with_background / len(mask_files) * 100:.2f}%)"
)

print(
    f"Tiles containing faults: "
    f"{tiles_with_faults} "
    f"({tiles_with_faults / len(mask_files) * 100:.2f}%)"
)

print(
    f"Tiles containing lava: "
    f"{tiles_with_lava} "
    f"({tiles_with_lava / len(mask_files) * 100:.2f}%)"
)

print(
    f"Tiles containing both faults and lava: "
    f"{tiles_with_faults_and_lava} "
    f"({tiles_with_faults_and_lava / len(mask_files) * 100:.2f}%)"
)

print(
    f"Tiles containing no foreground: "
    f"{tiles_with_no_foreground} "
    f"({tiles_with_no_foreground / len(mask_files) * 100:.2f}%)"
)


# ==========================================================
# FAULT COVERAGE WITHIN FAULT-CONTAINING TILES
# ==========================================================

# Store the percentage of each tile occupied by fault pixels.
# Tiles without any fault pixels are excluded from this analysis.
fault_coverages = []

for mask_path in mask_files:

    # Open the mask and convert it to a NumPy array.
    mask = np.array(Image.open(mask_path))

    # Count the number of pixels labelled as faults.
    fault_pixels = np.sum(mask == 128)

    # Only include tiles that actually contain faults.
    if fault_pixels > 0:

        # Calculate the percentage of the complete tile occupied
        # by the fault class.
        fault_percentage = (fault_pixels / mask.size) * 100
        fault_coverages.append(fault_percentage)

# Convert the list to a NumPy array so that summary statistics
# can be calculated.
fault_coverages = np.array(fault_coverages)


# ==========================================================
# PRINT FAULT COVERAGE STATISTICS
# ==========================================================

# These statistics describe how much of a typical
# fault-containing tile is occupied by the fault class.
print()
print("Fault coverage in fault-containing tiles:")
print()

print(f"Number of fault-containing tiles: {len(fault_coverages)}")
print(f"Minimum fault coverage: {fault_coverages.min():.2f}%")
print(f"25th percentile:        {np.percentile(fault_coverages, 25):.2f}%")
print(f"Median fault coverage:  {np.median(fault_coverages):.2f}%")
print(f"Mean fault coverage:    {fault_coverages.mean():.2f}%")
print(f"75th percentile:        {np.percentile(fault_coverages, 75):.2f}%")
print(f"Maximum fault coverage: {fault_coverages.max():.2f}%")


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
    f"{class_weights[128]:.4f}, "
    f"{class_weights[255]:.4f}"
    "]"
)