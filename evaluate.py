
# ==========================================================
# U-NET MODEL EVALUATION AND PREDICTION
# ==========================================================

# This script evaluates a trained U-Net model using paired
# SAR intensity and texture inputs.

# The model performs binary lava segmentation.

# The network produces one foreground output:

#     Channel 0 = Lava

# Pixels below the lava probability threshold are
# classified as non-lava.

# If ground-truth masks are available, lava IoU and Dice
# scores are calculated.

# If masks are unavailable, inference-only mode is used.

# The script:
# - Calculates lava probability maps.
# - Reconstructs overlapping tile predictions.
# - Displays the reconstructed segmentation.
# - Creates an overlay on the original SAR image.
# - Saves the final segmentation image.
# - Reads georeferencing from the original GeoTIFF.

# INPUT CHANNELS
# --------------
# Channel 0: SAR intensity, scaled to 0-1.
# Channel 1: Texture, original float32 ratios.

# ==========================================================


# ==========================================================
# IMPORTS
# ==========================================================

import random

import torch

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from utils.data_loading import LavaFlowDataset

from unet.unet_model import UNet

from PIL import Image

from pathlib import Path

import numpy as np

import rasterio

import geopandas as gpd

from shapely.geometry import LineString

from matplotlib.colors import ListedColormap

from datetime import datetime


# ==========================================================
# TILE POSITION FUNCTION
# ==========================================================

def parse_tile_position(filename):

    """
    Extract the x and y tile origin from a filename like:

        tile_0000_x0000_y0096.png

    These coordinates are used to reconstruct the
    complete testing image.
    """

    stem = filename.replace(".png", "")

    parts = stem.split("_")

    x_part = [
        part for part in parts
        if part.startswith("x")
    ][0]

    y_part = [
        part for part in parts
        if part.startswith("y")
    ][0]

    x = int(x_part[1:])

    y = int(y_part[1:])

    return x, y


# ==========================================================
# REFERENCE IMAGE LOADING
# ==========================================================

def load_reference_image(image_path):

    """
    Load the original image for plotting and return
    its CRS and affine transform when available.

    GeoTIFF:
        returns image_array, crs, transform

    Other images:
        returns image_array, None, None
    """

    image_path = Path(image_path)

    if image_path.suffix.lower() in {".tif", ".tiff"}:

        with rasterio.open(image_path) as src:

            image = src.read(1)

            crs = src.crs

            transform = src.transform

        return image, crs, transform

    image = Image.open(image_path).convert("L")

    return np.array(image), None, None


# ==========================================================
# SHAPEFILE EXPORT FUNCTION
# ==========================================================

def export_contour_shapefile(
    contour_set,
    transform,
    crs,
    output_path
):

    """
    Convert contour lines from pixel coordinates to
    map coordinates and save them as a shapefile.

    If CRS or transform is missing, export is skipped.
    """

    if transform is None or crs is None:

        print(
            "No CRS or transform found, "
            "skipping shapefile export."
        )

        return

    lines = []

    for segment in contour_set.allsegs[0]:

        if len(segment) < 2:
            continue

        world_coords = [
            transform * (x, y)
            for x, y in segment
        ]

        if len(world_coords) >= 2:

            lines.append(
                LineString(world_coords)
            )

    if not lines:

        print(
            "No contour segments found, "
            "skipping shapefile export."
        )

        return

    gdf = gpd.GeoDataFrame(
        {"level": [0.5] * len(lines)},
        geometry=lines,
        crs=crs,
    )

    gdf.to_file(output_path)

    print(f"Saved shapefile: {output_path}")


# ==========================================================
# EVALUATION SETTINGS
# ==========================================================

# Folder containing the tiled testing dataset.

TEST_DATASET = "testing"


# Original untiled testing image.
#
# Used to determine reconstruction dimensions and
# retrieve the original GeoTIFF georeferencing.
# Filename is defined in config.py.

from config import TESTING_IMAGE as TEST_IMAGE


# Folder for georeferenced shapefiles.

SHAPEFILE_OUTPUT_FOLDER = Path("data/shapefiles")

SHAPEFILE_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Number of testing tiles processed simultaneously.

TEST_BATCH_SIZE = 32





# Saved model checkpoint.

MODEL_PATH = Path("unet_lava_best.pth")


# Independent foreground probability thresholds.
#
# Lava probability threshold.
#
# Channel 0 = Lava

LAVA_THRESHOLD = 0.5


# Main folder containing archived training-run outputs.

MODEL_OUTPUT_FOLDER = Path("model_outputs")


# ==========================================================
# INTERSECTION AND UNION CALCULATION
# ==========================================================

def calculate_intersection_union(
    predictions,
    masks,
    num_classes=1
):

    """
    Calculate intersection and union for lava.

    predictions: [B, 1, H, W]
    masks:       [B, 1, H, W]
    """

    probabilities = torch.sigmoid(predictions)

    predicted_binary = torch.zeros_like(
        probabilities,
        dtype=torch.bool
    )

    predicted_binary[:, 0] = (
        probabilities[:, 0] >= LAVA_THRESHOLD
    )

    true_binary = masks > 0.5

    intersections = torch.zeros(
        num_classes,
        device=predictions.device
    )

    unions = torch.zeros(
        num_classes,
        device=predictions.device
    )

    for class_idx in range(num_classes):

        pred_class = predicted_binary[:, class_idx]

        true_class = true_binary[:, class_idx]

        intersections[class_idx] = (
            pred_class & true_class
        ).sum()

        unions[class_idx] = (
            pred_class | true_class
        ).sum()

    return intersections, unions


# ==========================================================
# DICE COMPONENT CALCULATION
# ==========================================================

def calculate_dice_components(
    predictions,
    masks,
    num_classes=1
):

    """
    Calculate Dice components for lava.

    predictions: [B, 1, H, W]
    masks:       [B, 1, H, W]
    """

    probabilities = torch.sigmoid(predictions)

    predicted_binary = torch.zeros_like(
        probabilities,
        dtype=torch.bool
    )

    predicted_binary[:, 0] = (
        probabilities[:, 0] >= LAVA_THRESHOLD
    )

    true_binary = masks > 0.5

    intersections = torch.zeros(
        num_classes,
        device=predictions.device
    )

    predicted_totals = torch.zeros(
        num_classes,
        device=predictions.device
    )

    true_totals = torch.zeros(
        num_classes,
        device=predictions.device
    )

    for class_idx in range(num_classes):

        pred_class = predicted_binary[:, class_idx]

        true_class = true_binary[:, class_idx]

        intersections[class_idx] = (
            pred_class & true_class
        ).sum()

        predicted_totals[class_idx] = pred_class.sum()

        true_totals[class_idx] = true_class.sum()

    return (
        intersections,
        predicted_totals,
        true_totals
    )

# ==========================================================
# INFERENCE-ONLY DATASET
# ==========================================================

class ImageOnlyTileDataset(Dataset):

    """
    Dataset for inference without ground-truth masks.

    Loads paired SAR and texture tiles.

    SAR:
        uint8 PNG, scaled to 0-1.

    Texture:
        float32 TIFF, original ratios preserved.

    Returns:
        image tensor [2, H, W]
        dummy mask tensor
        SAR tile filename
    """

    def __init__(self, images_dir, texture_dir):

        self.images_dir = Path(images_dir)

        self.texture_dir = Path(texture_dir)

        self.image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff"
            }
        )

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        # --------------------------------------------------
        # FIND MATCHING SAR AND TEXTURE FILES
        # --------------------------------------------------

        image_path = self.image_files[idx]

        texture_path = (
            self.texture_dir
            / f"{image_path.stem}.tif"
        )

        # --------------------------------------------------
        # LOAD SAR IMAGE
        # --------------------------------------------------

        image = Image.open(image_path).convert("L")

        image = (
            np.array(image, dtype=np.float32)
            / 255.0
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # --------------------------------------------------
        # LOAD TEXTURE MAP
        # --------------------------------------------------

        with rasterio.open(texture_path) as src:

            texture = src.read(1)

        texture = np.asarray(
            texture,
            dtype=np.float32
        )

        texture = torch.from_numpy(
            texture
        ).unsqueeze(0)

        # --------------------------------------------------
        # COMBINE INPUT CHANNELS
        # --------------------------------------------------

        # Channel 0 = SAR intensity
        # Channel 1 = Texture

        image = torch.cat(
            [image, texture],
            dim=0
        )

        # --------------------------------------------------
        # CREATE PLACEHOLDER MASK
        # --------------------------------------------------

        dummy_mask = torch.zeros_like(image)

        return image, dummy_mask, image_path.name


# ==========================================================
# CREATE TEST DATASET
# ==========================================================

images_dir = f"data/{TEST_DATASET}/images"

texture_dir = f"data/{TEST_DATASET}/texture"

masks_dir = Path(
    f"data/{TEST_DATASET}/masks"
)

has_masks = (
    masks_dir.exists()
    and any(masks_dir.iterdir())
)


# ----------------------------------------------------------
# TESTING WITH GROUND-TRUTH MASKS
# ----------------------------------------------------------

if has_masks:

    test_dataset = LavaFlowDataset(
        images_dir=images_dir,
        masks_dir=str(masks_dir),
        texture_dir=texture_dir
    )


# ----------------------------------------------------------
# TESTING WITHOUT GROUND-TRUTH MASKS
# ----------------------------------------------------------

else:

    test_dataset = ImageOnlyTileDataset(
        images_dir=images_dir,
        texture_dir=texture_dir
    )

    print(
        "No masks found, skipping IoU, Dice, "
        "and ground-truth validation plots."
    )


# ==========================================================
# CREATE TEST DATALOADER
# ==========================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

print(f"Testing images: {len(test_dataset)}")

print(
    f"Testing batches per epoch: {len(test_loader)}"
)


# ==========================================================
# LOAD TRAINED U-NET MODEL
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ==========================================================
# IDENTIFY TRAINING RUN
# ==========================================================

run_timestamp = checkpoint.get("run_timestamp")

if run_timestamp is None:

    raise KeyError(
        "The model checkpoint does not contain "
        "a run_timestamp."
    )

print(f"Model training run: {run_timestamp}")

RUN_FOLDER = MODEL_OUTPUT_FOLDER / run_timestamp

RUN_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

print(f"Model output folder: {RUN_FOLDER}")

model.eval()


# ==========================================================
# CALCULATE TEST METRICS
# ==========================================================

if has_masks:

    total_intersection = torch.zeros(
        1,
        device=device
    )

    total_union = torch.zeros(
        1,
        device=device
    )

    total_dice_intersection = torch.zeros(
        1,
        device=device
    )

    total_predicted = torch.zeros(
        1,
        device=device
    )

    total_true = torch.zeros(
        1,
        device=device
    )

    with torch.no_grad():

        for images, masks, filenames in test_loader:

            images = images.to(device)

            masks = masks.to(device)

            predictions = model(images)

            probabilities = torch.sigmoid(predictions)

            batch_intersection, batch_union = (
                calculate_intersection_union(
                    predictions,
                    masks
                )
            )

            total_intersection += batch_intersection

            total_union += batch_union

            (
                dice_intersection,
                batch_predicted,
                batch_true
            ) = calculate_dice_components(
                predictions,
                masks
            )

            total_dice_intersection += (
                dice_intersection
            )

            total_predicted += batch_predicted

            total_true += batch_true


    # ======================================================
    # CALCULATE IoU
    # ======================================================

    test_iou = torch.where(
        total_union > 0,
        total_intersection / total_union.clamp(min=1),
        torch.full_like(
            total_union,
            float("nan")
        )
    )

    lava_iou = test_iou[0].item()

    # ======================================================
    # CALCULATE DICE
    # ======================================================

    dice_denominator = (
        total_predicted + total_true
    )

    test_dice = torch.where(
        dice_denominator > 0,
        2.0 * total_dice_intersection
        / dice_denominator.clamp(min=1),
        torch.full_like(
            dice_denominator,
            float("nan")
        )
    )

    lava_dice = test_dice[0].item()


    # ======================================================
    # CALCULATE DICE
    # ======================================================

    dice_denominator = (
        total_predicted + total_true
    )

    test_dice = torch.where(
        dice_denominator > 0,
        2.0 * total_dice_intersection
        / dice_denominator.clamp(min=1),
        torch.full_like(
            dice_denominator,
            float("nan")
        )
    )

    faults_dice = test_dice[0].item()

    lava_dice = test_dice[1].item()

    valid_dice = test_dice[
        ~torch.isnan(test_dice)
    ]

    mean_foreground_dice = (
        valid_dice.mean().item()
        if valid_dice.numel() > 0
        else float("nan")
    )


    # ======================================================
    # PRINT TEST METRICS
    # ======================================================

    print()

    print("TEST SET EVALUATION")

    print("=" * 50)

    print(f"Lava IoU:         {lava_iou:.6f}")
    print(f"Lava Dice:        {lava_dice:.6f}")

    print("=" * 50)

else:

    print(
        "Validation metrics skipped because "
        "no ground-truth masks were found."
    )




# ==========================================================
# SEGMENTATION COLOUR MAP
# ==========================================================

class_cmap = ListedColormap([
    "#FAC6FE",  # Non-lava
    "#b30d66"   # Lava
])


# ==========================================================
# CREATE CLASS MAP
# ==========================================================

def create_class_map(probabilities):

    """
    Convert lava probabilities into a binary segmentation.

    0 = Non-lava
    1 = Lava
    """

    lava_prob = probabilities[:, 0]

    lava_detected = (
        lava_prob >= LAVA_THRESHOLD
    )

    class_map = torch.zeros_like(
        lava_prob,
        dtype=torch.long
    )

    class_map[lava_detected] = 1

    return class_map




# ==========================================================
# GENERATE PREDICTIONS FOR ALL TEST TILES
# ==========================================================

all_predicted_tiles = []

with torch.no_grad():

    for images, masks, filenames in test_loader:

        images = images.to(device)

        masks = masks.to(device)

        predictions = model(images)

        probabilities = torch.sigmoid(predictions)

        for i in range(probabilities.shape[0]):

            x, y = parse_tile_position(
                filenames[i]
            )

            all_predicted_tiles.append(
                {
                    "probability": probabilities[i].cpu(),
                    "x": x,
                    "y": y,
                    "filename": filenames[i],
                }
            )

print(
    f"Collected {len(all_predicted_tiles)} "
    "predicted tiles"
)


# ==========================================================
# LOAD ORIGINAL TESTING IMAGE
# ==========================================================

original_image, crs, transform = (
    load_reference_image(
        f"data/raw/images/{TEST_IMAGE}"
    )
)

image_height, image_width = (
    original_image.shape[:2]
)


# ==========================================================
# CREATE RECONSTRUCTION ARRAYS
# ==========================================================

probability_sum = torch.zeros(
    (1, image_height, image_width),
    dtype=torch.float32
)

count_map = torch.zeros(
    (image_height, image_width),
    dtype=torch.float32
)


# ==========================================================
# RECONSTRUCT OVERLAPPING TILE PREDICTIONS
# ==========================================================

for tile in all_predicted_tiles:

    probability_map = tile["probability"]

    x = tile["x"]

    y = tile["y"]

    _, tile_height, tile_width = (
        probability_map.shape
    )

    probability_sum[
        :,
        y:y + tile_height,
        x:x + tile_width
    ] += probability_map

    count_map[
        y:y + tile_height,
        x:x + tile_width
    ] += 1


# ==========================================================
# AVERAGE OVERLAPPING PREDICTIONS
# ==========================================================

safe_count_map = torch.clamp(
    count_map,
    min=1
).unsqueeze(0)

average_probability = (
    probability_sum / safe_count_map
)

final_mask = create_class_map(
    average_probability.unsqueeze(0)
).squeeze(0)


# ==========================================================
# DISPLAY STITCHED RECONSTRUCTION
# ==========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

# ----------------------------------------------------------
# LAVA PROBABILITY
# ----------------------------------------------------------

axes[0].imshow(
    average_probability[0],
    cmap="seismic",
    vmin=0,
    vmax=1
)

axes[0].set_title(
    "Lava probability map"
)

axes[0].axis("off")


# ----------------------------------------------------------
# FINAL SEGMENTATION
# ----------------------------------------------------------

axes[1].imshow(
    final_mask,
    cmap=class_cmap,
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Stitched segmentation"
)

axes[1].axis("off")

plt.tight_layout()

plt.show()



# ==========================================================
# PREPARE OUTPUT ARRAYS
# ==========================================================

test_image_name = Path(TEST_IMAGE).stem

evaluation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

final_mask_np = final_mask.cpu().numpy().astype(np.uint8)

lava_probability_np = (
    average_probability[0].cpu().numpy().astype(np.float32)
)

# ==========================================================
# SAVE FINAL SEGMENTATION GEOTIFF
# ==========================================================

if crs is not None and transform is not None:

    segmentation_tif_path = (
        RUN_FOLDER
        / f"{test_image_name}_{evaluation_timestamp}_predicted_segmentation.tif"
    )

    with rasterio.open(
        segmentation_tif_path,
        "w",
        driver="GTiff",
        height=image_height,
        width=image_width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:

        dst.write(final_mask_np, 1)

    print(f"Saved segmentation GeoTIFF: {segmentation_tif_path}")

else:
    print("No CRS or transform found, skipping GeoTIFF export.")


# ==========================================================
# SAVE FINAL SEGMENTATION PNG
# ==========================================================

segmentation_png_path = (
    RUN_FOLDER
    / f"{test_image_name}_{evaluation_timestamp}_predicted_segmentation_mask.png"
)

plt.imsave(
    segmentation_png_path,
    final_mask_np,
    cmap=class_cmap,
    vmin=0,
    vmax=1,
)

print(f"Saved segmentation PNG: {segmentation_png_path}")


# ==========================================================
# SAVE FULL-RESOLUTION PROBABILITY MAPS
# ==========================================================

probability_maps = {
    "lava": lava_probability_np,
}

for class_name, probability_map in probability_maps.items():

    # ------------------------------------------------------
    # SAVE PNG
    # ------------------------------------------------------

    png_path = (
        RUN_FOLDER
        / f"{test_image_name}_{evaluation_timestamp}_{class_name}_probability.png"
    )

    plt.imsave(
        png_path,
        probability_map,
        cmap="seismic",
        vmin=0,
        vmax=1,
    )

    print(f"Saved probability PNG: {png_path}")

    # ------------------------------------------------------
    # SAVE FLOAT32 GEOTIFF
    # ------------------------------------------------------

    if crs is not None and transform is not None:

        tif_path = (
            RUN_FOLDER
            / f"{test_image_name}_{evaluation_timestamp}_{class_name}_probability.tif"
        )

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=image_height,
            width=image_width,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:

            dst.write(probability_map, 1)

        print(f"Saved probability GeoTIFF: {tif_path}")


# ==========================================================
# NORMALISE ORIGINAL SAR IMAGE FOR DISPLAY
# ==========================================================

sar_image = np.asarray(original_image, dtype=np.float32)

valid_pixels = np.isfinite(sar_image)

if np.any(valid_pixels):

    sar_mean = np.mean(sar_image[valid_pixels])
    sar_std = np.std(sar_image[valid_pixels])

    lower_limit = sar_mean - 2.0 * sar_std
    upper_limit = sar_mean + 2.0 * sar_std

    if upper_limit > lower_limit:

        sar_normalised = np.clip(
            (sar_image - lower_limit)
            / (upper_limit - lower_limit),
            0,
            1,
        )

    else:
        sar_normalised = np.zeros_like(sar_image)

else:
    sar_normalised = np.zeros_like(sar_image)

sar_normalised[~valid_pixels] = 0


# ==========================================================
# CREATE SEGMENTATION OVERLAY
# ==========================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.imshow(
    sar_normalised,
    cmap="gray",
    vmin=0,
    vmax=1,
)

overlay = np.zeros(
    (image_height, image_width, 4),
    dtype=np.float32,
)




# ==========================================================
# LAVA OVERLAY
# ==========================================================

lava_pixels = final_mask_np == 1

overlay[lava_pixels] = [
    1.0,
    0.0,
    0.0,
    0.45,
]


# ==========================================================
# DISPLAY OVERLAY
# ==========================================================

ax.imshow(overlay)

ax.set_title("Predicted segmentation")

ax.axis("off")

plt.tight_layout()


# ==========================================================
# SAVE SEGMENTATION OVERLAY PNG
# ==========================================================

output_image_path = (
    RUN_FOLDER
    / f"{test_image_name}_{evaluation_timestamp}_predicted_segmentation.png"
)

fig.savefig(
    output_image_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved segmentation overlay: {output_image_path}")

plt.show()



# ==========================================================
# PRINT PREDICTED CLASS COUNTS
# ==========================================================

unique, counts = torch.unique(
    final_mask,
    return_counts=True
)

print(
    dict(
        zip(
            unique.tolist(),
            counts.tolist()
        )
    )
)


# ==========================================================
# PRINT CLASS COUNTS FROM FINAL TILE BATCH
# ==========================================================

final_batch_probabilities = (
    torch.sigmoid(predictions)
)

final_batch_classes = create_class_map(
    final_batch_probabilities
)

unique, counts = torch.unique(
    final_batch_classes,
    return_counts=True
)

class_names = {
    0: "Non-lava",
    1: "Lava"
}

print()

print("Final tile batch class counts:")

for class_idx, count in zip(
    unique.tolist(),
    counts.tolist()
):

    print(
        f"{class_names[class_idx]}: "
        f"{count:,} pixels"
    )
