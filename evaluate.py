# ==========================================================
# U-NET MODEL EVALUATION AND PREDICTION
# ==========================================================
#
# This script evaluates a trained three-class U-Net model on a
# tiled testing dataset and reconstructs the predictions into
# the dimensions of the original testing image.
#
# The model uses three segmentation classes:
#   0 = Background
#   1 = Faults
#   2 = Lava
#
# If ground-truth masks are available, the script calculates
# IoU and Dice scores for each class.
#
# If masks are not available, the script can still run in
# inference-only mode and generate predictions.
#
# The script also:
#   - Displays randomly selected test tiles and their predictions.
#   - Calculates class probability maps.
#   - Reconstructs overlapping tile predictions into one image.
#   - Displays the reconstructed segmentation over the original image.
#   - Saves the final segmentation image in the corresponding
#     model training-run folder.
#   - Loads georeferencing information when the original image
#     is a GeoTIFF.


# ==========================================================
# IMPORTS
# ==========================================================

# Used to randomly select test tiles for visual inspection.
import random

# PyTorch is used to load the model, perform inference, and
# manipulate prediction tensors.
import torch

# Matplotlib is used to display predictions and probability maps.
import matplotlib.pyplot as plt

# DataLoader is used to process test images in batches.
from torch.utils.data import DataLoader

# Custom dataset class used when both images and masks are available.
from utils.data_loading import LavaFlowDataset

# U-Net architecture used during training.
from unet.unet_model import UNet

# PIL is used to open image files.
from PIL import Image

# Path is used for file and folder paths.
from pathlib import Path

# NumPy is used for image arrays and the final segmentation overlay.
import numpy as np

# Rasterio is used to read GeoTIFF images and their georeferencing.
import rasterio

# GeoPandas and LineString are used for exporting georeferenced
# contour lines as shapefiles.
import geopandas as gpd
from shapely.geometry import LineString

from PIL import Image


# ==========================================================
# TILE POSITION FUNCTION
# ==========================================================

def parse_tile_position(filename):

    # Extract the x and y tile origin from a filename like:
    # tile_0000_x0000_y0096.png
    #
    # These coordinates are required later to place each tile
    # prediction back into its original position when reconstructing
    # the complete testing image.

    stem = filename.replace(".png", "")
    parts = stem.split("_")

    # Find the filename components containing the x and y origins.
    x_part = [part for part in parts if part.startswith("x")][0]
    y_part = [part for part in parts if part.startswith("y")][0]

    # Remove the x/y prefix and convert the remaining value to an integer.
    x = int(x_part[1:])
    y = int(y_part[1:])

    return x, y


# ==========================================================
# REFERENCE IMAGE LOADING
# ==========================================================

def load_reference_image(image_path):
    """
    Load the original image for plotting and, if available,
    return its CRS and affine transform.

    GeoTIFF:
        returns image_array, crs, transform

    PNG or other non-georeferenced image:
        returns image_array, None, None
    """

    # Convert the supplied location to a Path object.
    image_path = Path(image_path)

    # GeoTIFF images may contain coordinate reference system and
    # affine-transform information, so they are opened with rasterio.
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(image_path) as src:
            image = src.read(1)  # first band as a 2D array
            crs = src.crs
            transform = src.transform
        return image, crs, transform

    # Non-georeferenced images are opened with PIL and converted
    # to greyscale to match the imagery used by the model.
    image = Image.open(image_path).convert("L")
    return np.array(image), None, None


# ==========================================================
# SHAPEFILE EXPORT FUNCTION
# ==========================================================

def export_contour_shapefile(contour_set, transform, crs, output_path):
    """
    Convert the contour line from pixel coordinates to map coordinates
    and save it as a shapefile.

    If CRS or transform is missing, export is skipped.
    """

    # A georeferenced shapefile cannot be produced without both
    # the coordinate reference system and affine transform.
    if transform is None or crs is None:
        print("No CRS or transform found, skipping shapefile export.")
        return

    # Store the converted contour segments.
    lines = []

    # levels=[0.5] gives one contour level, so allsegs[0] is the segment list
    for segment in contour_set.allsegs[0]:

        # A line requires at least two coordinate positions.
        if len(segment) < 2:
            continue

        # segment[:, 0] = x (column), segment[:, 1] = y (row)
        #
        # Convert each contour coordinate from image pixel coordinates
        # to the map coordinates defined by the GeoTIFF transform.
        world_coords = [transform * (x, y) for x, y in segment]

        if len(world_coords) >= 2:
            lines.append(LineString(world_coords))

    # Do not create an empty shapefile if no valid lines were found.
    if not lines:
        print("No contour segments found, skipping shapefile export.")
        return

    # Store the line geometries in a georeferenced GeoDataFrame.
    gdf = gpd.GeoDataFrame(
        {"level": [0.5] * len(lines)},
        geometry=lines,
        crs=crs,
    )

    # Write the GeoDataFrame to disk as a shapefile.
    gdf.to_file(output_path)
    print(f"Saved shapefile: {output_path}")


# ==========================================================
# EVALUATION SETTINGS
# ==========================================================

# Folder containing the tiled test dataset.
#
# For example, "testing" corresponds to:
#   data/testing/images
#   data/testing/masks
TEST_DATASET = "testing"

# Original, untiled testing image.
#
# This image is used to determine the dimensions of the complete
# reconstruction and to display the final segmentation.
#
# This can be a standard image or a GeoTIFF.
TEST_IMAGE = "new_testingim.tif"

# Folder where georeferenced shapefiles will be written.
SHAPEFILE_OUTPUT_FOLDER = Path("data/shapefiles")

# Create the folder if it does not already exist.
SHAPEFILE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Number of testing tiles processed simultaneously.
TEST_BATCH_SIZE = 32

# Number of randomly selected test tiles displayed for visual inspection.
PLOT_EXAMPLES = 4

# Saved checkpoint from the two-output U-Net.
MODEL_PATH = Path("unet_2output_best.pth")

# Independent probability thresholds.
# Channel 0 = Faults
# Channel 1 = Lava
FAULT_THRESHOLD = 0.5
LAVA_THRESHOLD = 0.5

# Main folder containing the archived outputs from training runs.
MODEL_OUTPUT_FOLDER = Path("model_outputs")


# ==========================================================
# INTERSECTION AND UNION CALCULATION
# ==========================================================

def calculate_intersection_union(predictions, masks, num_classes=2):
    """
    Calculate intersection and union independently for
    faults and lava.

    Channel 0 = Faults
    Channel 1 = Lava

    predictions: [B, 2, H, W]
    masks:       [B, 2, H, W]
    """

    # Convert raw logits into independent probabilities.
    probabilities = torch.sigmoid(predictions)

    # Apply the foreground probability thresholds.
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

    # Convert ground-truth masks to Boolean values.
    true_binary = masks > 0.5

    # Initialise intersection and union for both classes.
    intersections = torch.zeros(
        num_classes,
        device=predictions.device
    )

    unions = torch.zeros(
        num_classes,
        device=predictions.device
    )

    # Calculate IoU independently for faults and lava.
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

def calculate_dice_components(predictions, masks, num_classes=2):
    """
    Calculate Dice components independently for faults and lava.

    Channel 0 = Faults
    Channel 1 = Lava

    predictions: [B, 2, H, W]
    masks:       [B, 2, H, W]
    """

    # Convert raw logits into independent probabilities.
    probabilities = torch.sigmoid(predictions)

    # Apply the foreground probability thresholds.
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

    # Convert ground-truth masks to Boolean values.
    true_binary = masks > 0.5

    # Initialise accumulators for both foreground classes.
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

    # Calculate Dice components separately for faults and lava.
    for class_idx in range(num_classes):

        pred_class = predicted_binary[:, class_idx]
        true_class = true_binary[:, class_idx]

        # Pixels correctly predicted as this class.
        intersections[class_idx] = (
            pred_class & true_class
        ).sum()

        # Total pixels predicted as this class.
        predicted_totals[class_idx] = pred_class.sum()

        # Total ground-truth pixels belonging to this class.
        true_totals[class_idx] = true_class.sum()

    return intersections, predicted_totals, true_totals


# ==========================================================
# INFERENCE-ONLY DATASET
# ==========================================================

# Dataset is also imported here for the image-only dataset class.
from torch.utils.data import DataLoader, Dataset


class ImageOnlyTileDataset(Dataset):
    """
    Dataset for inference only.
    Returns an image tensor, a dummy mask tensor, and the filename.

    This allows the evaluation pipeline to run on images for which
    manually labelled ground-truth masks are not available.
    """

    def __init__(self, images_dir):

        # Store the image directory as a Path object.
        self.images_dir = Path(images_dir)

        # Find supported image files and sort them so that they
        # are processed in a consistent order.
        self.image_files = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )

    def __len__(self):

        # Return the number of image tiles in the dataset.
        return len(self.image_files)

    def __getitem__(self, idx):

        # Find the image corresponding to the requested dataset index.
        image_path = self.image_files[idx]

        # Open the image in greyscale.
        image = Image.open(image_path).convert("L")

        # Convert the image to a floating-point NumPy array and
        # scale pixel values from 0-255 to 0-1.
        image = np.array(image, dtype=np.float32) / 255.0

        # Convert to a PyTorch tensor and add the single image
        # channel expected by the U-Net.
        image = torch.from_numpy(image).unsqueeze(0)

        # Create a placeholder mask so that the returned data have
        # the same basic structure as the supervised dataset.
        dummy_mask = torch.zeros_like(image)

        return image, dummy_mask, image_path.name


# ==========================================================
# CREATE TEST DATASET
# ==========================================================

# Define the folders containing the tiled testing images and masks.
images_dir = f"data/{TEST_DATASET}/images"
masks_dir = Path(f"data/{TEST_DATASET}/masks")

# Determine whether ground-truth masks are available.
has_masks = masks_dir.exists() and any(masks_dir.iterdir())

# If masks exist, use the normal supervised dataset so that
# quantitative evaluation metrics can be calculated.
if has_masks:
    test_dataset = LavaFlowDataset(
        images_dir=images_dir,
        masks_dir=str(masks_dir),
    )

# If masks do not exist, use the image-only dataset and perform
# inference without quantitative validation.
else:
    test_dataset = ImageOnlyTileDataset(images_dir=images_dir)
    print("No masks found, skipping IoU, Dice, and ground-truth validation plots.")


# ==========================================================
# CREATE TEST DATALOADER
# ==========================================================

# Create the test DataLoader.
#
# shuffle=False preserves the dataset ordering, which is useful
# when reconstructing predictions from tile filenames.
test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

# Report the size of the test dataset.
print(f"Testing images: {len(test_dataset)}")
print(f"Testing batches per epoch: {len(test_loader)}")


# ==========================================================
# LOAD TRAINED U-NET MODEL
# ==========================================================

# Use the GPU if CUDA is available, otherwise use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the U-Net architecture and move it to the selected device.
model = UNet().to(device)

# Load the saved model checkpoint.
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Restore the trained model parameters.
model.load_state_dict(checkpoint["model_state_dict"])


# ==========================================================
# IDENTIFY TRAINING RUN
# ==========================================================

# Read the timestamp of the training run from the active model.
#
# This links the active model checkpoint to its corresponding
# archived folder within model_outputs.
run_timestamp = checkpoint.get("run_timestamp")

if run_timestamp is None:
    raise KeyError(
        "The model checkpoint does not contain a run_timestamp."
    )

print(f"Model training run: {run_timestamp}")

RUN_FOLDER = MODEL_OUTPUT_FOLDER / run_timestamp

RUN_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"Model output folder: {RUN_FOLDER}")

# Put the model into evaluation mode.
model.eval()


# ==========================================================
# CALCULATE TEST METRICS
# ==========================================================

# Only calculate metrics when ground-truth masks are available.
if has_masks:

    # Accumulate intersection and union for faults and lava.
    total_intersection = torch.zeros(2, device=device)
    total_union = torch.zeros(2, device=device)

    # Accumulate components needed to calculate Dice.
    total_dice_intersection = torch.zeros(2, device=device)
    total_predicted = torch.zeros(2, device=device)
    total_true = torch.zeros(2, device=device)

    # Disable gradient calculations during evaluation.
    with torch.no_grad():

        # Process the complete testing dataset.
        for images, masks, filenames in test_loader:

            # Move images and masks to the selected device.
            images = images.to(device)
            masks = masks.to(device)

            # Generate raw predictions.
            predictions = model(images)

            # Convert logits to independent probabilities.
            probabilities = torch.sigmoid(predictions)

            # Calculate intersection and union.
            batch_intersection, batch_union = calculate_intersection_union(
                predictions,
                masks
            )

            total_intersection += batch_intersection
            total_union += batch_union

            # Calculate Dice components.
            dice_intersection, batch_predicted, batch_true = (
                calculate_dice_components(
                    predictions,
                    masks
                )
            )

            total_dice_intersection += dice_intersection
            total_predicted += batch_predicted
            total_true += batch_true

    # ======================================================
    # CALCULATE IoU
    # ======================================================

    # Avoid division by zero when a class is absent.
    test_iou = torch.where(
        total_union > 0,
        total_intersection / total_union.clamp(min=1),
        torch.full_like(total_union, float("nan"))
    )

    # Extract IoU for each foreground class.
    faults_iou = test_iou[0].item()
    lava_iou = test_iou[1].item()

    # Calculate mean IoU across classes with defined scores.
    valid_iou = test_iou[~torch.isnan(test_iou)]

    mean_foreground_iou = (
        valid_iou.mean().item()
        if valid_iou.numel() > 0
        else float("nan")
    )

    # ======================================================
    # CALCULATE DICE
    # ======================================================

    dice_denominator = total_predicted + total_true

    test_dice = torch.where(
        dice_denominator > 0,
        2.0 * total_dice_intersection / dice_denominator.clamp(min=1),
        torch.full_like(dice_denominator, float("nan"))
    )

    # Extract Dice for each foreground class.
    faults_dice = test_dice[0].item()
    lava_dice = test_dice[1].item()

    # Calculate mean Dice across classes with defined scores.
    valid_dice = test_dice[~torch.isnan(test_dice)]

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

# Select up to PLOT_EXAMPLES random tiles from the test dataset.
# If the dataset contains fewer tiles, display all available tiles.
plot_examples = min(PLOT_EXAMPLES, len(test_dataset))
plot_indices = random.sample(range(len(test_dataset)), plot_examples)

plot_images = []
plot_masks = []

# Load the selected tiles.
for idx in plot_indices:
    image, mask, filename = test_dataset[idx]
    plot_images.append(image)
    plot_masks.append(mask)

# Combine the selected images and masks into batches and move
# them to the selected device.
images = torch.stack(plot_images).to(device)
masks = torch.stack(plot_masks).to(device)

# Generate predictions for the selected example tiles.
with torch.no_grad():

    # Generate raw model predictions.
    predictions = model(images)

    # Convert logits into independent probabilities.
    # Channel 0 = Faults
    # Channel 1 = Lava
    probabilities = torch.sigmoid(predictions)

    # Apply independent probability thresholds.
    predicted_faults = (
        probabilities[:, 0] >= FAULT_THRESHOLD
    )

    predicted_lava = (
        probabilities[:, 1] >= LAVA_THRESHOLD
    )
# ==========================================================
# SEGMENTATION COLOUR MAP
# ==========================================================

from matplotlib.colors import ListedColormap

# Colours used to display the final segmentation.
#
# 0 = Background (dark grey)
# 1 = Faults (blue)
# 2 = Lava (red)
#
# Background is not a learned output channel.
# It is assigned when neither foreground probability
# exceeds its threshold.

class_cmap = ListedColormap([
    "#FAC6FE",  # Background
    "#00bebe",  # Faults
    "#b30d66"   # Lava
])


# ==========================================================
# DISPLAY EXAMPLE TILES
# ==========================================================

# Convert the two independent foreground predictions into
# a single segmentation map for visualisation.
#
# Final labels:
# 0 = Background
# 1 = Faults
# 2 = Lava
#
# If both thresholds are exceeded, assign the pixel to
# whichever foreground class has the higher probability.

def create_class_map(probabilities):

    # Extract independent foreground probabilities.
    fault_prob = probabilities[:, 0]
    lava_prob = probabilities[:, 1]

    # Apply the configured thresholds.
    fault_detected = fault_prob >= FAULT_THRESHOLD
    lava_detected = lava_prob >= LAVA_THRESHOLD

    # Initialise all pixels as background.
    class_map = torch.zeros_like(
        fault_prob,
        dtype=torch.long
    )

    # Assign fault pixels.
    class_map[fault_detected] = 1

    # Assign lava pixels where the fault threshold was not met,
    # or where lava has the higher probability.
    lava_selected = lava_detected & (
        ~fault_detected | (lava_prob > fault_prob)
    )

    class_map[lava_selected] = 2

    return class_map


# Create predicted segmentation maps for the example tiles.
predicted_classes = create_class_map(probabilities)


if has_masks:

    # Display five columns:
    # 1. Input image
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

        image = images[i].cpu().squeeze(0)

        # Convert the two binary ground-truth channels
        # into a three-label map for display.
        true_mask = torch.zeros_like(
            masks[i, 0],
            dtype=torch.long
        )

        true_mask[masks[i, 0] > 0.5] = 1
        true_mask[masks[i, 1] > 0.5] = 2

        true_mask = true_mask.cpu()

        predicted_class = predicted_classes[i].cpu()

        fault_probability = probabilities[i, 0].cpu()
        lava_probability = probabilities[i, 1].cpu()

        axes[i][0].imshow(image, cmap="gray")
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
        axes[i][2].set_title("Predicted segmentation")

        axes[i][3].imshow(
            fault_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )
        axes[i][3].set_title("Fault probability")

        axes[i][4].imshow(
            lava_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )
        axes[i][4].set_title("Lava probability")

        for ax in axes[i]:
            ax.axis("off")

else:

    # Without ground-truth masks, display four columns:
    # Input, prediction, fault probability, lava probability.

    fig, axes = plt.subplots(
        plot_examples,
        4,
        figsize=(16, 4 * plot_examples),
        squeeze=False
    )

    for i in range(plot_examples):

        image = images[i].cpu().squeeze(0)
        predicted_class = predicted_classes[i].cpu()

        fault_probability = probabilities[i, 0].cpu()
        lava_probability = probabilities[i, 1].cpu()

        axes[i][0].imshow(image, cmap="gray")
        axes[i][0].set_title("Input image")

        axes[i][1].imshow(
            predicted_class,
            cmap=class_cmap,
            vmin=0,
            vmax=2
        )
        axes[i][1].set_title("Predicted segmentation")

        axes[i][2].imshow(
            fault_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )
        axes[i][2].set_title("Fault probability")

        axes[i][3].imshow(
            lava_probability,
            cmap="seismic",
            vmin=0,
            vmax=1
        )
        axes[i][3].set_title("Lava probability")

        for ax in axes[i]:
            ax.axis("off")


plt.tight_layout()
plt.show()


# ==========================================================
# GENERATE PREDICTIONS FOR ALL TEST TILES
# ==========================================================

# Store the probability maps and original positions of all tiles
# so that they can later be reconstructed into one complete image.
all_predicted_tiles = []

# Run inference across the complete testing dataset.
with torch.no_grad():
    for images, masks, filenames in test_loader:

        # Move the current batch to the selected device.
        images = images.to(device)
        masks = masks.to(device)

        # Generate raw model predictions.
        predictions = model(images)

        # Convert logits into independent probabilities.
        #
        # Channel 0 = Faults
        # Channel 1 = Lava
        #
        # Sigmoid calculates each foreground probability independently.
        # The probabilities do not need to sum to 1.

        probabilities = torch.sigmoid(predictions)

        # Save each probability volume, along with its location.
        for i in range(probabilities.shape[0]):

            # Read the original x and y position from the tile filename.
            x, y = parse_tile_position(filenames[i])

            # Store the two channel probability map and the tile position.
            all_predicted_tiles.append(
                {
                    "probability": probabilities[i].cpu(),
                    "x": x,
                    "y": y,
                    "filename": filenames[i],
                }
            )

# Check we have the expected number of tiles
print(f"Collected {len(all_predicted_tiles)} predicted tiles")


# ==========================================================
# LOAD ORIGINAL TESTING IMAGE
# ==========================================================

# Open the original testing image to determine the reconstruction size.
#
# If the image is a GeoTIFF, its CRS and affine transform are
# also returned.
original_image, crs, transform = load_reference_image(
    f"data/raw/images/{TEST_IMAGE}"
)

# Get the height and width of the original image.
image_height, image_width = original_image.shape[:2]


# ==========================================================
# CREATE RECONSTRUCTION ARRAYS
# ==========================================================

# Create a blank probability volume matching the original image.
#
# The model produces two independent foreground channels:
# Channel 0 = Faults
# Channel 1 = Lava
#
# Background does not have a probability channel.

probability_sum = torch.zeros(
    (2, image_height, image_width),
    dtype=torch.float32
)

# Record how many overlapping tiles contribute predictions
# to each pixel in the reconstructed image.

count_map = torch.zeros(
    (image_height, image_width),
    dtype=torch.float32
)

# ==========================================================
# RECONSTRUCT OVERLAPPING TILE PREDICTIONS
# ==========================================================

# Add each predicted probability map into the reconstruction buffers.
for tile in all_predicted_tiles:

    # Probability map contains two foreground channels.
    # Channel 0 = Faults
    # Channel 1 = Lava

    probability_map = tile["probability"]  # shape: [2, H, W]

    # Original position of the tile within the complete image.
    x = tile["x"]
    y = tile["y"]

    # Determine the dimensions of the current tile.
    _, tile_height, tile_width = probability_map.shape

    # Add the probability values to their original image position.
    probability_sum[:, y:y + tile_height, x:x + tile_width] += probability_map

    # Record that another tile contributes to these pixels.
    count_map[y:y + tile_height, x:x + tile_width] += 1


# ==========================================================
# AVERAGE OVERLAPPING PREDICTIONS
# ==========================================================

# Prevent division by zero in pixels that have no contributing tiles.
safe_count_map = torch.clamp(count_map, min=1).unsqueeze(0)

# Average the overlapping probability values.
#
# Where tiles overlap, each tile therefore contributes equally
# to the final class probabilities.
average_probability = probability_sum / safe_count_map

# Convert the two averaged foreground probability maps
# into a final segmentation containing three labels.
#
# 0 = Background
# 1 = Faults
# 2 = Lava
#
# Background is assigned when neither foreground
# probability exceeds its threshold.
#
# If both thresholds are exceeded, the class with
# the higher probability is selected.

final_mask = create_class_map(
    average_probability.unsqueeze(0)
).squeeze(0)


# ==========================================================
# DISPLAY STITCHED RECONSTRUCTION
# ==========================================================

# Display the reconstructed foreground probability maps
# and the final threshold-based segmentation.

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Display the reconstructed lava probability map.
# Channel 1 = Lava
axes[0].imshow(
    average_probability[1],
    cmap="seismic",
    vmin=0,
    vmax=1
)
axes[0].set_title("Lava probability map")
axes[0].axis("off")

# Display the final segmentation.
#
# 0 = Background
# 1 = Faults
# 2 = Lava
#
# Background is assigned where neither foreground
# probability exceeds its threshold.
axes[1].imshow(
    final_mask,
    cmap=class_cmap,
    vmin=0,
    vmax=2
)
axes[1].set_title("Stitched segmentation")
axes[1].axis("off")

# Display the reconstructed fault probability map.
# Channel 0 = Faults
axes[2].imshow(
    average_probability[0],
    cmap="seismic",
    vmin=0,
    vmax=1
)
axes[2].set_title("Fault probability map")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# ==========================================================
# CREATE SEGMENTATION OVERLAY
# ==========================================================

# Display the stitched prediction as a semi-transparent mask
# over the original image.
fig, ax = plt.subplots(figsize=(12, 8))

# Show the original testing image in greyscale.
ax.imshow(original_image, cmap="gray")

# Convert the final prediction to a NumPy array.
final_mask_np = final_mask.numpy()

# Create an RGBA overlay.
#
# Shape is [height, width, 4].
# The first three channels contain red, green, and blue values.
# The fourth channel controls transparency.
overlay = np.zeros(
    (image_height, image_width, 4),
    dtype=np.float32
)


#
# Display class 1 in semi-transparent blue.
fault_pixels = final_mask_np == 1
overlay[fault_pixels] = [0.0, 0.4, 1.0, 0.45]   # blue, 45% opacity

# Class 2 = lava
#
# Display class 2 in semi-transparent red.
lava_pixels = final_mask_np == 2
overlay[lava_pixels] = [1.0, 0.0, 0.0, 0.45]    # red, 45% opacity

# Class 0 = background remains completely transparent.
ax.imshow(overlay)

ax.set_title("Predicted segmentation")
ax.axis("off")

plt.tight_layout()


# ==========================================================
# SAVE SEGMENTATION IMAGE
# ==========================================================

# Use the original testing image filename, without its extension,
# when naming the saved prediction.
test_image_name = Path(TEST_IMAGE).stem

# Save the image inside the archived folder belonging to the
# training run that produced the active model.
output_image_path = (
    RUN_FOLDER
    / f"{test_image_name}_predicted_segmentation.png"
)

# Save the figure at high resolution.
fig.savefig(
    output_image_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved segmentation image: {output_image_path}")

# Display the final segmentation overlay.
plt.show()


# ==========================================================
# PRINT PREDICTED CLASS COUNTS
# ==========================================================

# Count how many pixels in the complete reconstructed image
# were assigned to each segmentation class.
unique, counts = torch.unique(final_mask, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))


# ==========================================================
# PRINT CLASS COUNTS FROM FINAL TILE BATCH
# ==========================================================

# Convert the final batch of raw model predictions
# into independent foreground probabilities.
#
# Channel 0 = Faults
# Channel 1 = Lava

final_batch_probabilities = torch.sigmoid(predictions)

# Apply the same threshold-based classification used
# for the complete reconstructed image.
#
# 0 = Background
# 1 = Faults
# 2 = Lava

final_batch_classes = create_class_map(
    final_batch_probabilities
)

# Count the number of pixels assigned to each class.
unique, counts = torch.unique(
    final_batch_classes,
    return_counts=True
)

# Print the class counts.
class_names = {
    0: "Background",
    1: "Faults",
    2: "Lava"
}

print()
print("Final tile batch class counts:")

for class_idx, count in zip(unique.tolist(), counts.tolist()):
    print(f"{class_names[class_idx]}: {count:,} pixels")