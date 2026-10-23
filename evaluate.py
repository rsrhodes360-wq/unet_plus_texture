
"""
==========================================================
U-NET MODEL EVALUATION AND PREDICTION
==========================================================

This script evaluates a trained U-Net model using paired
SAR intensity and texture inputs.

The model uses three segmentation classes:

    0 = Background
    1 = Faults
    2 = Lava

The network produces two independent foreground outputs:

    Channel 0 = Faults
    Channel 1 = Lava

If ground-truth masks are available, IoU and Dice scores
are calculated.

If masks are unavailable, inference-only mode is used.

The script:
- Displays randomly selected test tiles and predictions.
- Calculates class probability maps.
- Reconstructs overlapping tile predictions.
- Displays the reconstructed segmentation.
- Creates an overlay on the original SAR image.
- Saves the final segmentation image.
- Reads georeferencing from the original GeoTIFF.

INPUT CHANNELS
--------------
Channel 0: SAR intensity, scaled to 0-1.
Channel 1: Texture, original float32 ratios.

==========================================================
"""

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

TEST_IMAGE = "testing_image.tif"


# Folder for georeferenced shapefiles.

SHAPEFILE_OUTPUT_FOLDER = Path("data/shapefiles")

SHAPEFILE_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# Number of testing tiles processed simultaneously.

TEST_BATCH_SIZE = 32


# Number of randomly selected test tiles displayed.

PLOT_EXAMPLES = 4


# Saved model checkpoint.

MODEL_PATH = Path("unet_2output_best.pth")


# Independent foreground probability thresholds.
#
# Channel 0 = Faults
# Channel 1 = Lava

FAULT_THRESHOLD = 0.5

LAVA_THRESHOLD = 0.5


# Main folder containing archived training-run outputs.

MODEL_OUTPUT_FOLDER = Path("model_outputs")


# ==========================================================
# INTERSECTION AND UNION CALCULATION
# ==========================================================

def calculate_intersection_union(
    predictions,
    masks,
    num_classes=2
):

    """
    Calculate intersection and union independently
    for faults and lava.

    predictions: [B, 2, H, W]
    masks:       [B, 2, H, W]
    """

    probabilities = torch.sigmoid(predictions)

    predicted_binary = torch.zeros_like(
        probabilities,
        dtype=torch.bool
    )

    predicted_binary[:, 0] = (
        probabilities[:, 0] >= FAULT_THRESHOLD
    )

    predicted_binary[:, 1] = (
        probabilities[:, 1] >= LAVA_THRESHOLD
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
    num_classes=2
):

    """
    Calculate Dice components independently
    for faults and lava.

    predictions: [B, 2, H, W]
    masks:       [B, 2, H, W]
    """

    probabilities = torch.sigmoid(predictions)

    predicted_binary = torch.zeros_like(
        probabilities,
        dtype=torch.bool
    )

    predicted_binary[:, 0] = (
        probabilities[:, 0] >= FAULT_THRESHOLD
    )

    predicted_binary[:, 1] = (
        probabilities[:, 1] >= LAVA_THRESHOLD
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
        2,
        device=device
    )

    total_union = torch.zeros(
        2,
        device=device
    )

    total_dice_intersection = torch.zeros(
        2,
        device=device
    )

    total_predicted = torch.zeros(
        2,
        device=device
    )

    total_true = torch.zeros(
        2,
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

    faults_iou = test_iou[0].item()

    lava_iou = test_iou[1].item()

    valid_iou = test_iou[
        ~torch.isnan(test_iou)
    ]

    mean_foreground_iou = (
        valid_iou.mean().item()
        if valid_iou.numel() > 0
        else float("nan")
    )


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

    print(f"Faults IoU:       {faults_iou:.6f}")
    print(f"Lava IoU:         {lava_iou:.6f}")
    print(f"Mean FG IoU:      {mean_foreground_iou:.6f}")

    print(f"Faults Dice:      {faults_dice:.6f}")
    print(f"Lava Dice:        {lava_dice:.6f}")
    print(f"Mean FG Dice:     {mean_foreground_dice:.6f}")

    print("=" * 50)

else:

    print(
        "Validation metrics skipped because "
        "no ground-truth masks were found."
    )


# ==========================================================
# SELECT RANDOM TEST TILES FOR VISUAL INSPECTION
# ==========================================================

plot_examples = min(
    PLOT_EXAMPLES,
    len(test_dataset)
)

plot_indices = random.sample(
    range(len(test_dataset)),
    plot_examples
)

plot_images = []

plot_masks = []

for idx in plot_indices:

    image, mask, filename = test_dataset[idx]

    plot_images.append(image)

    plot_masks.append(mask)

images = torch.stack(
    plot_images
).to(device)

masks = torch.stack(
    plot_masks
).to(device)


# ==========================================================
# GENERATE EXAMPLE PREDICTIONS
# ==========================================================

with torch.no_grad():

    predictions = model(images)

    probabilities = torch.sigmoid(predictions)

    predicted_faults = (
        probabilities[:, 0] >= FAULT_THRESHOLD
    )

    predicted_lava = (
        probabilities[:, 1] >= LAVA_THRESHOLD
    )


# ==========================================================
# SEGMENTATION COLOUR MAP
# ==========================================================

class_cmap = ListedColormap([
    "#FAC6FE",  # Background
    "#00bebe",  # Faults
    "#b30d66"   # Lava
])


# ==========================================================
# CREATE CLASS MAP
# ==========================================================

def create_class_map(probabilities):

    """
    Convert independent foreground probabilities
    into a single three-label segmentation.

    0 = Background
    1 = Faults
    2 = Lava

    If both thresholds are exceeded, assign the
    class with the higher probability.
    """

    fault_prob = probabilities[:, 0]

    lava_prob = probabilities[:, 1]

    fault_detected = (
        fault_prob >= FAULT_THRESHOLD
    )

    lava_detected = (
        lava_prob >= LAVA_THRESHOLD
    )

    class_map = torch.zeros_like(
        fault_prob,
        dtype=torch.long
    )

    class_map[fault_detected] = 1

    lava_selected = lava_detected & (
        ~fault_detected
        | (lava_prob > fault_prob)
    )

    class_map[lava_selected] = 2

    return class_map


predicted_classes = create_class_map(
    probabilities
)


# ==========================================================
# DISPLAY EXAMPLE TILES
# ==========================================================

if has_masks:

    # Five columns:
    #
    # 1. SAR input
    # 2. Ground truth
    # 3. Predicted segmentation
    # 4. Fault probability
    # 5. Lava probability

    fig, axes = plt.subplots(
        plot_examples,
        5,
        figsize=(20, 4 * plot_examples),
        squeeze=False
    )

    for i in range(plot_examples):

        # Display SAR channel 0 only.
        #
        # Texture channel 1 is still supplied to the
        # model but is not displayed in this plot.

        image = images[i, 0].cpu()

        true_mask = torch.zeros_like(
            masks[i, 0],
            dtype=torch.long
        )

        true_mask[masks[i, 0] > 0.5] = 1

        true_mask[masks[i, 1] > 0.5] = 2

        true_mask = true_mask.cpu()

        predicted_class = (
            predicted_classes[i].cpu()
        )

        fault_probability = (
            probabilities[i, 0].cpu()
        )

        lava_probability = (
            probabilities[i, 1].cpu()
        )

        axes[i][0].imshow(
            image,
            cmap="gray"
        )

        axes[i][0].set_title("Input image")

        axes[i][1].imshow(
            true_mask,
            cmap=class_cmap,
            vmin=0,
            vmax=2
        )

        axes[i][1].set_title("Ground truth")

        axes[i][2].imshow(
            predicted_class,
            cmap=class_cmap,
            vmin=0,
            vmax=2
        )

        axes[i][2].set_title(
            "Predicted segmentation"
        )

        axes[i][3].imshow(
            fault_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )

        axes[i][3].set_title(
            "Fault probability"
        )

        axes[i][4].imshow(
            lava_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )

        axes[i][4].set_title(
            "Lava probability"
        )

        for ax in axes[i]:

            ax.axis("off")


else:

    # Four columns:
    #
    # 1. SAR input
    # 2. Predicted segmentation
    # 3. Fault probability
    # 4. Lava probability

    fig, axes = plt.subplots(
        plot_examples,
        4,
        figsize=(16, 4 * plot_examples),
        squeeze=False
    )

    for i in range(plot_examples):

        # Display SAR channel 0 only.

        image = images[i, 0].cpu()

        predicted_class = (
            predicted_classes[i].cpu()
        )

        fault_probability = (
            probabilities[i, 0].cpu()
        )

        lava_probability = (
            probabilities[i, 1].cpu()
        )

        axes[i][0].imshow(
            image,
            cmap="gray"
        )

        axes[i][0].set_title("Input image")

        axes[i][1].imshow(
            predicted_class,
            cmap=class_cmap,
            vmin=0,
            vmax=2
        )

        axes[i][1].set_title(
            "Predicted segmentation"
        )

        axes[i][2].imshow(
            fault_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )

        axes[i][2].set_title(
            "Fault probability"
        )

        axes[i][3].imshow(
            lava_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )

        axes[i][3].set_title(
            "Lava probability"
        )

        for ax in axes[i]:

            ax.axis("off")


plt.tight_layout()

plt.show()


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
    (2, image_height, image_width),
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
    3,
    figsize=(18, 6)
)

# ----------------------------------------------------------
# LAVA PROBABILITY
# ----------------------------------------------------------

axes[0].imshow(
    average_probability[1],
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
    vmax=2
)

axes[1].set_title(
    "Stitched segmentation"
)

axes[1].axis("off")


# ----------------------------------------------------------
# FAULT PROBABILITY
# ----------------------------------------------------------

axes[2].imshow(
    average_probability[0],
    cmap="seismic",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Fault probability map"
)

axes[2].axis("off")

plt.tight_layout()

plt.show()


# ==========================================================
# CREATE SEGMENTATION OVERLAY
# ==========================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)

ax.imshow(
    original_image,
    cmap="gray"
)

final_mask_np = final_mask.numpy()

overlay = np.zeros(
    (image_height, image_width, 4),
    dtype=np.float32
)


# ----------------------------------------------------------
# FAULTS
# ----------------------------------------------------------

fault_pixels = final_mask_np == 1

overlay[fault_pixels] = [
    0.0,
    0.4,
    1.0,
    0.45
]


# ----------------------------------------------------------
# LAVA
# ----------------------------------------------------------

lava_pixels = final_mask_np == 2

overlay[lava_pixels] = [
    1.0,
    0.0,
    0.0,
    0.45
]

# Background remains transparent.

ax.imshow(overlay)

ax.set_title(
    "Predicted segmentation"
)

ax.axis("off")

plt.tight_layout()


# ==========================================================
# SAVE SEGMENTATION IMAGE
# ==========================================================

test_image_name = Path(
    TEST_IMAGE
).stem

output_image_path = (
    RUN_FOLDER
    / f"{test_image_name}_predicted_segmentation.png"
)

fig.savefig(
    output_image_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Saved segmentation image: {output_image_path}"
)

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
    0: "Background",
    1: "Faults",
    2: "Lava"
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
