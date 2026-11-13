
"""
==========================================================
SAR TEXTURE MAP GENERATION
==========================================================

Generates locally normalised SAR texture maps using a
21 x 21 pixel rectangular focal mean filter.

Texture = Original SAR / Local mean SAR

Input:
    data/raw/images/

Output:
    data/raw/texture/

The original SAR images are not modified.

Texture maps are exported as float32 GeoTIFFs without
min-max normalisation or conversion to 8-bit.

NoData pixels are excluded from the focal mean.
Invalid output pixels are stored as NaN.

==========================================================
"""

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, label, find_objects


# ==========================================================
# SETTINGS
# ==========================================================

INPUT_FOLDER = Path("data/raw/images")
OUTPUT_FOLDER = Path("data/raw/texture")

from config import (
    TRAINING_IMAGE,
    VALIDATION_IMAGE,
    TESTING_IMAGE,
    NEW_TRAINING_IMAGES,
    NEW_VALIDATION_IMAGES,
    NEW_TESTING_IMAGES,
)


KERNEL_SIZE = 21


# Minimum number of 8-connected zero-valued SAR pixels
# required for a region to be classified as NoData.

MIN_NODATA_REGION_SIZE = 6

# Reflect values at image boundaries when calculating
# the focal mean.
BOUNDARY_MODE = "reflect"


# ==========================================================
# GENERATE TEXTURE MAP
# ==========================================================

def generate_texture(image_name):

    input_path = INPUT_FOLDER / image_name

    output_path = (
        OUTPUT_FOLDER
        / f"{Path(image_name).stem}_texture.tif"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"SAR image not found: {input_path}"
        )

    print(f"\nProcessing: {image_name}")

    # ------------------------------------------------------
    # LOAD ORIGINAL SAR IMAGE
    # ------------------------------------------------------

    with rasterio.open(input_path) as src:

        if src.count != 1:
            raise ValueError(
                f"Expected a single-band SAR image: {input_path}"
            )

        # Read the original pixel values as float32 for
        # mathematical processing.
        sar = src.read(1).astype(np.float32)

        # Read the valid-data mask from the GeoTIFF.
        valid = src.read_masks(1) > 0

        # Preserve the original GeoTIFF metadata.
        profile = src.profile.copy()

    valid &= np.isfinite(sar)



    # ------------------------------------------------------
    # IDENTIFY ZERO-VALUED NODATA REGIONS
    # ------------------------------------------------------

    # Identify zero-valued pixels in the original SAR image.
    # Only consider pixels not already marked as NoData.

    zero_pixels = (sar == 0) & valid

    # Define 8-connectivity, including diagonal neighbours.

    connectivity = np.ones((3, 3), dtype=np.uint8)

    # Identify connected regions of zero-valued pixels.

    labelled_regions, number_of_regions = label(
        zero_pixels,
        structure=connectivity
    )

    # Count the number of pixels in each connected region.
    # Region 0 represents pixels outside the zero-valued regions.

    region_sizes = np.bincount(labelled_regions.ravel())

    # Identify connected zero-valued regions that are
    # large enough to be considered NoData.

    nodata_region_ids = np.where(
        region_sizes >= MIN_NODATA_REGION_SIZE
    )[0]

    # Exclude background label 0.

    nodata_region_ids = nodata_region_ids[
        nodata_region_ids != 0
    ]

    # Identify the bounding box of each connected region.

    region_slices = find_objects(labelled_regions)

    # Retain regions that are exactly one pixel wide.

    for region_id in nodata_region_ids.copy():

        region_slice = region_slices[region_id - 1]

        if region_slice is None:
            continue

        region_height = (
            region_slice[0].stop - region_slice[0].start
        )

        region_width = (
            region_slice[1].stop - region_slice[1].start
        )

        if (region_width == 1) != (region_height == 1):
            nodata_region_ids = nodata_region_ids[
                nodata_region_ids != region_id
            ]

    # Classify the remaining regions as NoData.

    nodata_regions = np.isin(
        labelled_regions,
        nodata_region_ids
    )

    # Exclude these regions from the valid-data array.

    valid &= ~nodata_regions

    # Report the classification results.

    newly_invalid_pixels = np.count_nonzero(nodata_regions)

    retained_zero_pixels = np.count_nonzero(
        zero_pixels & ~nodata_regions
    )

    print(
        f"Zero-valued SAR regions found: {number_of_regions:,}"
    )

    print(
        f"Pixels classified as additional NoData: "
        f"{newly_invalid_pixels:,}"
    )

    print(
        f"Zero-valued pixels retained as valid: "
        f"{retained_zero_pixels:,}"
    )




    # ------------------------------------------------------
    # CALCULATE 21 x 21 FOCAL MEAN
    # ------------------------------------------------------

    # Replace invalid pixels with zero for the summation.
    sar_valid = np.where(valid, sar, 0.0)

    # Calculate the local sum of valid SAR values.
    local_sum = uniform_filter(
        sar_valid,
        size=KERNEL_SIZE,
        mode=BOUNDARY_MODE,
        output=np.float64,
    )

    # Calculate the local proportion of valid pixels.
    local_count = uniform_filter(
        valid.astype(np.float64),
        size=KERNEL_SIZE,
        mode=BOUNDARY_MODE,
        output=np.float64,
    )

    # Divide the local sum by the valid-pixel proportion.
    # The common kernel-area factor cancels.
    focal_mean = np.full(
        sar.shape,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        local_sum,
        local_count,
        out=focal_mean,
        where=local_count > 0,
    )

    # ------------------------------------------------------
    # CALCULATE TEXTURE RATIO
    # ------------------------------------------------------

    texture = np.full(
        sar.shape,
        np.nan,
        dtype=np.float32,
    )

    # Prevent division by zero and exclude invalid pixels.
    valid_output = (
        valid
        & np.isfinite(focal_mean)
        & (focal_mean != 0)
    )

    np.divide(
        sar,
        focal_mean,
        out=texture,
        where=valid_output,
    )

    # ------------------------------------------------------
    # PREPARE OUTPUT GEOTIFF
    # ------------------------------------------------------

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # SAVE TEXTURE MAP
    # ------------------------------------------------------

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(texture, 1)

    print(f"Saved: {output_path}")

    valid_values = texture[np.isfinite(texture)]

    if valid_values.size > 0:
        print(f"Minimum ratio: {valid_values.min():.6f}")
        print(f"Maximum ratio: {valid_values.max():.6f}")
        print(f"Mean ratio:    {valid_values.mean():.6f}")
    else:
        print("Warning: No valid texture values generated.")


# ==========================================================
# DATASET SELECTION
# ==========================================================

def ask_yes_no(question):

    while True:

        answer = input(
            f"{question} (y/n): "
        ).strip().lower()

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter y or n.")


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print("\nSAR TEXTURE MAP GENERATION")
    print("=" * 50)

    datasets = [
        ("training", TRAINING_IMAGE),
        ("validation", VALIDATION_IMAGE),
        ("testing", TESTING_IMAGE),
    ]

    for i, image_name in enumerate(NEW_TRAINING_IMAGES, start=1):
        datasets.append(
            (f"additional training {i}", image_name)
        )

    for image_name in NEW_VALIDATION_IMAGES:
        datasets.append(
            (f"additional validation {Path(image_name).stem}", image_name)
        )

    for image_name in NEW_TESTING_IMAGES:
        datasets.append(
            (f"additional testing {Path(image_name).stem}", image_name)
        )

    selected = []

    for dataset_name, image_name in datasets:

        if ask_yes_no(
            f"Generate {dataset_name} texture map?"
        ):
            selected.append(image_name)

    if not selected:
        print("\nNo datasets selected.")
        return

    for image_name in selected:
        generate_texture(image_name)

    print("\nTexture map generation complete.")


if __name__ == "__main__":
    main()
