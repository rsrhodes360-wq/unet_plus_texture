
"""
==========================================================
DATASET TILING SCRIPT - SAR AND TEXTURE
==========================================================

Prepares aligned SAR, texture and segmentation mask tiles
for the U-Net model.

SAR images:
    data/raw/images/

Texture maps:
    data/raw/texture/

Segmentation masks:
    data/raw/masks/

Outputs:
    data/training/
    data/validation/
    data/testing/

SAR tiles are saved as uint8 PNG files.
Texture tiles are saved as float32 TIFF files.
Mask tiles are saved as PNG files.

Only selected datasets are processed.

Existing output folders are preserved. Only the files
inside the selected dataset folders are deleted.

Mask values:
    0   = Background
    128 = Faults
    255 = Lava

==========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path

from PIL import Image
import numpy as np
import rasterio


# ==========================================================
# FOLDER LOCATIONS
# ==========================================================

INPUT_FOLDER = Path("data/raw")

IMAGE_FOLDER = INPUT_FOLDER / "images"

TEXTURE_FOLDER = INPUT_FOLDER / "texture"

MASK_FOLDER = INPUT_FOLDER / "masks"


# ==========================================================
# TILE SETTINGS
# ==========================================================

TILE_SIZE = 256

OVERLAP = 0.25

STRIDE = int(TILE_SIZE * (1 - OVERLAP))


# ==========================================================
# DATASETS
# ==========================================================

# Dataset filenames are defined in config.py.

from config import (
    TRAINING_IMAGE,
    TRAINING_TEXTURE,
    TRAINING_MASK,
    VALIDATION_IMAGE,
    VALIDATION_TEXTURE,
    VALIDATION_MASK,
    TESTING_IMAGE,
    TESTING_TEXTURE,
    TESTING_MASK,
)


# ==========================================================
# CREATE REQUIRED INPUT FOLDERS
# ==========================================================

def ensure_required_folders():

    """
    Make sure the input folder structure exists.
    """

    INPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    TEXTURE_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    MASK_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# ASK USER WHICH DATASETS TO TILE
# ==========================================================

def ask_yes_no(question):

    """
    Ask the user to enter y or n.
    """

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
# CALCULATE TILE POSITIONS
# ==========================================================

def calculate_positions(image_length):

    """
    Calculate tile starting positions along one dimension.

    The final tile is positioned against the image boundary.
    """

    positions = []

    position = 0

    while position + TILE_SIZE <= image_length:

        positions.append(position)

        position += STRIDE

    if positions[-1] != image_length - TILE_SIZE:

        positions.append(image_length - TILE_SIZE)

    return positions


# ==========================================================
# OPEN INPUT IMAGE
# ==========================================================

def open_image_any(image_path):

    """
    Open PNG, JPG, JPEG, BMP, TIFF or GeoTIFF images.

    This retains the original image-loading behaviour.

    Standard images are opened using PIL.

    TIFF and GeoTIFF images are opened using rasterio
    and converted into PIL images.
    """

    suffix = image_path.suffix.lower()

    if suffix in [".png", ".jpg", ".jpeg", ".bmp"]:

        return Image.open(image_path)

    if suffix in [".tif", ".tiff"]:

        with rasterio.open(image_path) as src:

            array = src.read(1)

            return Image.fromarray(array)

    raise ValueError(
        f"Unsupported image format: {image_path.suffix}"
    )


# ==========================================================
# OPEN TEXTURE MAP
# ==========================================================

def open_texture_map(texture_path):

    """
    Load a float32 texture map.

    Texture maps are stored as TIFF or GeoTIFF files.

    No normalisation, scaling or conversion to uint8
    is performed.
    """

    if not texture_path.is_file():

        raise FileNotFoundError(
            f"Texture map not found: {texture_path}"
        )

    with rasterio.open(texture_path) as src:

        texture = src.read(1)

    return texture


# ==========================================================
# LOAD IMAGE, TEXTURE AND MASK
# ==========================================================

def load_dataset(
    image_name,
    texture_name,
    mask_name=None
):

    """
    Load the original SAR image, texture map and
    optional segmentation mask.

    Images and masks retain their original loading
    and cleaning behaviour.

    Corresponding pixel coordinates are used for tiling.
    """

    # ------------------------------------------------------
    # LOAD SAR IMAGE
    # ------------------------------------------------------

    image_path = IMAGE_FOLDER / image_name

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = open_image_any(image_path)

    # ------------------------------------------------------
    # LOAD TEXTURE MAP
    # ------------------------------------------------------

    texture_path = TEXTURE_FOLDER / texture_name

    texture = open_texture_map(texture_path)

    # ------------------------------------------------------
    # CHECK SAR AND TEXTURE DIMENSIONS
    # ------------------------------------------------------

    if texture.shape != (image.height, image.width):

        raise ValueError(
            "SAR and texture dimensions do not match.\n"
            f"SAR: {image.width} x {image.height}\n"
            f"Texture: {texture.shape[1]} x "
            f"{texture.shape[0]}"
        )

    # ------------------------------------------------------
    # LOAD OPTIONAL SEGMENTATION MASK
    # ------------------------------------------------------

    mask = None

    if mask_name is not None:

        mask_path = MASK_FOLDER / mask_name

        if mask_path.is_file():

            mask = open_image_any(mask_path)

            # Clean the mask using the original method.
            #
            # 0   = Background
            # 128 = Faults
            # 255 = Lava
            #
            # All other values become background.

            mask = mask.point(
                lambda p:
                128 if p == 128
                else 255 if p == 255
                else 0
            )

        else:

            print(
                f"No mask found for '{image_name}'. "
                "Image tiles will be created without "
                "mask tiles."
            )

    return image, texture, mask


# ==========================================================
# PREPARE OUTPUT FOLDERS
# ==========================================================

def prepare_output_folder(base_folder):

    """
    Empty the contents of the selected dataset folder.

    The directory structure itself is never deleted.

    Files in images, texture and masks are removed.

    Files in unselected datasets are not touched.
    """

    base_folder = Path(base_folder)

    base_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for subfolder in ["images", "texture", "masks"]:

        folder = base_folder / subfolder

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for item in folder.rglob("*"):

            if item.is_file() or item.is_symlink():

                item.unlink()


# ==========================================================
# SAVE INDIVIDUAL TILE
# ==========================================================

def save_tile(
    image_tile,
    texture_tile,
    mask_tile,
    filename,
    output_folder
):

    """
    Save corresponding SAR, texture and mask tiles.

    SAR:
        PNG

    Texture:
        Float32 TIFF

    Mask:
        PNG, if available

    All tiles use the same filename stem.
    """

    base_folder = Path("data") / output_folder

    # ------------------------------------------------------
    # SAVE SAR IMAGE TILE
    # ------------------------------------------------------

    image_tile.save(
        base_folder / "images" / filename
    )

    # ------------------------------------------------------
    # SAVE TEXTURE TILE
    # ------------------------------------------------------

    texture_filename = (
        Path(filename).stem + ".tif"
    )

    texture_path = (
        base_folder
        / "texture"
        / texture_filename
    )

    texture_tile = np.asarray(
        texture_tile,
        dtype=np.float32
    )

    with rasterio.open(
        texture_path,
        "w",
        driver="GTiff",
        height=texture_tile.shape[0],
        width=texture_tile.shape[1],
        count=1,
        dtype="float32",
        nodata=np.nan
    ) as dst:

        dst.write(texture_tile, 1)

    # ------------------------------------------------------
    # SAVE OPTIONAL MASK TILE
    # ------------------------------------------------------

    if mask_tile is not None:

        mask_tile.save(
            base_folder / "masks" / filename
        )


# ==========================================================
# TILE ONE DATASET
# ==========================================================

def process_dataset(
    image_name,
    texture_name,
    mask_name,
    output_folder,
    start_index=0
):

    """
    Tile one SAR image, corresponding texture map and
    optional segmentation mask.

    The same coordinates are used for all three.

    start_index allows multiple training areas to be
    combined without repeating tile numbers.
    """

    # ------------------------------------------------------
    # LOAD DATASET
    # ------------------------------------------------------

    image, texture, mask = load_dataset(
        image_name,
        texture_name,
        mask_name
    )

    # ------------------------------------------------------
    # CALCULATE TILE POSITIONS
    # ------------------------------------------------------

    x_positions = calculate_positions(image.width)

    y_positions = calculate_positions(image.height)

    # ------------------------------------------------------
    # INITIALISE TILE NUMBERING
    # ------------------------------------------------------

    tile_count = start_index

    # ------------------------------------------------------
    # GENERATE TILES
    # ------------------------------------------------------

    for y in y_positions:

        for x in x_positions:

            # ------------------------------------------------
            # CROP SAR IMAGE
            # ------------------------------------------------

            image_tile = image.crop(
                (
                    x,
                    y,
                    x + TILE_SIZE,
                    y + TILE_SIZE
                )
            )

            # ------------------------------------------------
            # CROP TEXTURE MAP
            # ------------------------------------------------

            texture_tile = texture[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            # ------------------------------------------------
            # CROP OPTIONAL MASK
            # ------------------------------------------------

            if mask is not None:

                mask_tile = mask.crop(
                    (
                        x,
                        y,
                        x + TILE_SIZE,
                        y + TILE_SIZE
                    )
                )

            else:

                mask_tile = None

            # ------------------------------------------------
            # CREATE TILE FILENAME
            # ------------------------------------------------

            filename = (
                f"tile_{tile_count:04d}"
                f"_x{x:04d}"
                f"_y{y:04d}.png"
            )

            # ------------------------------------------------
            # SAVE MATCHING TILES
            # ------------------------------------------------

            save_tile(
                image_tile,
                texture_tile,
                mask_tile,
                filename,
                output_folder
            )

            tile_count += 1

    # ------------------------------------------------------
    # REPORT NUMBER OF TILES
    # ------------------------------------------------------

    print(
        f"{output_folder}: saved "
        f"{tile_count - start_index} tiles "
        f"from {image_name}."
    )


# ==========================================================
# FIND NEXT AVAILABLE TILE NUMBER
# ==========================================================

def get_next_tile_index(output_folder):

    """
    Find the next unused tile number.

    This preserves the original ability to combine
    multiple training areas in one dataset.
    """

    images_folder = (
        Path("data")
        / output_folder
        / "images"
    )

    if not images_folder.exists():

        return 0

    max_index = -1

    for file in images_folder.glob("tile_*.png"):

        stem = file.stem

        parts = stem.split("_")

        if len(parts) >= 4 and parts[1].isdigit():

            try:

                tile_index = int(parts[1])

                max_index = max(
                    max_index,
                    tile_index
                )

            except ValueError:

                pass

    return max_index + 1


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print()
    print("=" * 50)
    print("SAR AND TEXTURE DATASET TILING")
    print("=" * 50)

    # ------------------------------------------------------
    # CREATE REQUIRED INPUT FOLDERS
    # ------------------------------------------------------

    ensure_required_folders()

    # ------------------------------------------------------
    # ASK WHICH DATASETS TO TILE
    # ------------------------------------------------------

    tile_training = ask_yes_no(
        "Do you want to tile the training dataset?"
    )

    tile_validation = ask_yes_no(
        "Do you want to tile the validation dataset?"
    )

    tile_testing = ask_yes_no(
        "Do you want to tile the testing dataset?"
    )

    if not any(
        [tile_training, tile_validation, tile_testing]
    ):

        print("\nNo datasets selected.")

        return

    # ======================================================
    # TRAINING DATA
    # ======================================================

    if tile_training:

        # Empty only the training output folders.
        #
        # The folders themselves are preserved.

        prepare_output_folder("data/training")

        # Tile the first training area.

        process_dataset(
            image_name=TRAINING_IMAGE,
            texture_name=TRAINING_TEXTURE,
            mask_name=TRAINING_MASK,
            output_folder="training",
            start_index=0
        )

        '''
        # --------------------------------------------------
        # OPTIONAL SECOND TRAINING AREA
        # --------------------------------------------------

        # Find the next unused training tile number.

        next_training_index = get_next_tile_index(
            "training"
        )

        # Tile the second training area and append its
        # tiles to the same training folder.

        process_dataset(
            image_name=TRAINING_IMAGE_2,
            texture_name=TRAINING_TEXTURE_2,
            mask_name=TRAINING_MASK_2,
            output_folder="training",
            start_index=next_training_index
        )
        '''

    else:

        print(
            "\nTraining skipped. "
            "Existing tiles preserved."
        )

    # ======================================================
    # VALIDATION DATA
    # ======================================================

    if tile_validation:

        prepare_output_folder("data/validation")

        process_dataset(
            image_name=VALIDATION_IMAGE,
            texture_name=VALIDATION_TEXTURE,
            mask_name=VALIDATION_MASK,
            output_folder="validation",
            start_index=0
        )

    else:

        print(
            "\nValidation skipped. "
            "Existing tiles preserved."
        )

    # ======================================================
    # TESTING DATA
    # ======================================================

    if tile_testing:

        prepare_output_folder("data/testing")

        process_dataset(
            image_name=TESTING_IMAGE,
            texture_name=TESTING_TEXTURE,
            mask_name=TESTING_MASK,
            output_folder="testing",
            start_index=0
        )

    else:

        print(
            "\nTesting skipped. "
            "Existing tiles preserved."
        )

    print()
    print("=" * 50)
    print("TILING COMPLETE")
    print("=" * 50)


# ==========================================================
# RUN SCRIPT
# ==========================================================

if __name__ == "__main__":

    main()
