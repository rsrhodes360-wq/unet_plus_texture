
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
from scipy.ndimage import uniform_filter


# ==========================================================
# SETTINGS
# ==========================================================

INPUT_FOLDER = Path("data/raw/images")
OUTPUT_FOLDER = Path("data/raw/texture")

from config import TRAINING_IMAGE, VALIDATION_IMAGE, TESTING_IMAGE
KERNEL_SIZE = 21

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
