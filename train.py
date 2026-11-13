#  ==========================================================
#  U-NET TRAINING SCRIPT
#  ==========================================================
#
#  This script trains a vanilla U-Net for binary lava
#  segmentation of radar imagery.
#
#  The model produces one foreground output:
#
#    Channel 0 = Lava
#
#  Pixels not labelled as lava have a target value of zero.
#
#  The script:
#
#    - Loads the tiled training and validation datasets.
#    - Optionally applies data augmentation to the training data.
#    - Uses binary cross-entropy and Dice loss during training.
#    - Calculates lava IoU on the validation dataset.
#    - Uses lava IoU to assess validation performance.
#    - Reduces the learning rate if validation performance stops
#      improving.
#    - Saves a checkpoint whenever lava IoU improves.
#    - Creates a separate folder and Excel training log for each run.
#    - Can optionally resume training from an existing checkpoint.
#
#  Before training begins, the script displays the main training
#  settings and asks the user to confirm that the dataset and class
#  weights have been checked.
#
#  Training data are expected in:
#
#    data/training/images
#    data/training/masks
#
#  Validation data are expected in:
#
#    data/validation/images
#    data/validation/masks

#  ==========================================================

#  IMPORTS

#  ==========================================================



#  os is used to check whether an existing model checkpoint is

#  available when resuming training.

import os

from statistics import pstdev

#  PyTorch provides the model training and tensor operations.

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



#  DataLoader loads the dataset in batches during training

#  and validation.

from torch.utils.data import DataLoader



#  ReduceLROnPlateau reduces the learning rate when validation

#  performance stops improving.

from torch.optim.lr_scheduler import ReduceLROnPlateau



#  Custom dataset class used to load the tiled images and masks.

from utils.data_loading import LavaFlowDataset



#  U-Net architecture used for segmentation.

from unet.unet_model import UNet



#  Used to create a unique timestamp for each training run.

from datetime import datetime



#  Path is used for model-output file and folder locations.

from pathlib import Path



#  openpyxl is used to create the Excel training log.

from openpyxl import Workbook

from openpyxl.styles import Font, PatternFill, Alignment



import matplotlib.pyplot as plt





#  ==========================================================

#  TRAINING SETTINGS

#  ==========================================================



#  These are the main settings that can be changed between

#  training experiments.



#  Number of image tiles processed together during each

#  optimisation step.

BATCH_SIZE = 8

NUM_WORKERS = 2
PIN_MEMORY = True

#  Maximum number of complete passes through the training dataset.

NUM_EPOCHS = 200



#  Initial learning rate used by the Adam optimiser.

LEARNING_RATE = 1e-4


# Standard deviation penalty for checkpoint selection.
# Higher values place greater emphasis on consistent
# validation IoU across geographical areas.

IOU_STD_PENALTY = 0.5


#  Set to True to continue training from MODEL_PATH.

#  Set to False to initialise a new model and train from scratch.

RESUME_TRAINING = False



#  Main model checkpoint used by evaluate.py and optionally used

#  to resume training.

MODEL_PATH = "unet_lava_best.pth"



#  Positive-class weight for the lava output.
#
#  Channel 0 = Lava
#
#  This is an initial value, npythonot a calculated class weight.
#  We will assess class imbalance before selecting the final weight.

POS_WEIGHTS = [1]



#  Set to True to apply the geometric augmentation implemented

#  by LavaFlowDataset to the training dataset.

#

#  Augmentation is not applied to the validation dataset.

USE_AUGMENTATION = True
USE_BRIGHTNESS_AUGMENTATION = True






#  ==========================================================

#  FUNCTIONS

#  ==========================================================


#  ==========================================================
#  IDENTIFY VALIDATION AREA FROM TILE FILENAME
#  ==========================================================

def get_validation_area(filename):

    """
    Extract the validation area identifier from a tile filename.

    area1_tile_0000_x0000_y0000.png -> area1
    area2_tile_0000_x0000_y0000.png -> area2
    """

    area = filename.split("_")[0]

    if not area.startswith("area") or not area[4:].isdigit():
        raise ValueError(
            f"Invalid validation tile filename: {filename}"
        )

    return area


#  ==========================================================
#  INTERSECTION AND UNION CALCULATION
#  ==========================================================

def calculate_intersection_union(predictions, masks, threshold=0.5):
    """
    Calculate intersection and union for lava.

    Channel 0 = Lava

    predictions: [B, 1, H, W]
    masks:       [B, 1, H, W]
    """

    # Convert raw logits into probabilities.
    probabilities = torch.sigmoid(predictions)

    # Apply the probability threshold.
    predicted_binary = probabilities >= threshold

    # Convert ground-truth masks to Boolean values.
    true_binary = masks > 0.5

    # Initialise intersection and union for lava.
    intersections = torch.zeros(
        1,
        device=predictions.device
    )

    unions = torch.zeros(
        1,
        device=predictions.device
    )

    # Calculate IoU components for lava.
    for class_idx in range(1):

        pred_class = predicted_binary[:, class_idx]
        true_class = true_binary[:, class_idx]

        # Pixels correctly predicted as lava.
        intersections[class_idx] = (
            pred_class & true_class
        ).sum()

        # Pixels predicted or labelled as lava.
        unions[class_idx] = (
            pred_class | true_class
        ).sum()

    return intersections, unions



#  ==========================================================
#  BINARY DICE LOSS
#  ==========================================================

def dice_loss_binary(logits, targets, eps=1e-6):
    """
    Calculate Dice loss for lava.

    Channel 0 = Lava

    logits:  [B, 1, H, W]
    targets: [B, 1, H, W]
    """

    # Convert logits into probabilities.
    probs = torch.sigmoid(logits)

    # Sum over batch and spatial dimensions.
    dims = (0, 2, 3)

    # Calculate intersection for lava.
    intersection = torch.sum(probs * targets, dim=dims)

    # Calculate total predicted and ground-truth values.
    denominator = torch.sum(probs + targets, dim=dims)

    # Calculate Dice for lava.
    dice_per_class = (
        (2.0 * intersection + eps)
        / (denominator + eps)
    )

    # Return lava Dice loss.
    return 1.0 - dice_per_class.mean()


#  ==========================================================
#  VALIDATION FUNCTION
#  ==========================================================

def evaluate(model, data_loader, criterion_bce, device):

    """
    Evaluate the model on all validation areas.

    Calculates intersection and union for all tiles
    in each batch simultaneously.

    Returns:
        average_loss
        overall_iou
        area_ious
        mean_area_iou
    """

    model.eval()

    running_loss = 0.0

    total_intersection = torch.zeros(1, device=device)
    total_union = torch.zeros(1, device=device)

    area_intersections = {}
    area_unions = {}

    with torch.no_grad():

        for images, masks, filenames in data_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            # Preserve the existing validation loss.

            bce_loss = criterion_bce(predictions, masks)
            d_loss = dice_loss_binary(predictions, masks)

            loss = bce_loss + d_loss

            running_loss += loss.item()

            # Calculate predictions for the entire batch.

            probabilities = torch.sigmoid(predictions)

            predicted_binary = probabilities >= 0.5
            true_binary = masks > 0.5

            # Calculate intersection and union for every
            # tile simultaneously.
            #
            # Result: one value per tile in the batch.

            tile_intersections = (
                predicted_binary & true_binary
            ).sum(dim=(1, 2, 3))

            tile_unions = (
                predicted_binary | true_binary
            ).sum(dim=(1, 2, 3))

            # Accumulate overall intersection and union.

            total_intersection += tile_intersections.sum()
            total_union += tile_unions.sum()

            # Identify which validation areas occur
            # in the current batch.

            batch_areas = [
                get_validation_area(filename)
                for filename in filenames
            ]

            # Accumulate intersection and union by area.
            #
            # Tiles from the same area are grouped together,
            # rather than processed individually.

            for area in set(batch_areas):

                area_indices = [
                    i for i, tile_area in enumerate(batch_areas)
                    if tile_area == area
                ]

                if area not in area_intersections:

                    area_intersections[area] = torch.zeros(
                        1, device=device
                    )

                    area_unions[area] = torch.zeros(
                        1, device=device
                    )

                area_intersections[area] += (
                    tile_intersections[area_indices].sum()
                )

                area_unions[area] += (
                    tile_unions[area_indices].sum()
                )

    # Preserve the existing average validation loss.

    average_loss = running_loss / len(data_loader)

    # Calculate overall pixel-pooled IoU.

    overall_iou = torch.where(
        total_union > 0,
        total_intersection / total_union.clamp(min=1),
        torch.full_like(total_union, float("nan"))
    )

    # Calculate individual regional IoUs.

    area_ious = {}

    for area in sorted(
        area_intersections,
        key=lambda name: int(name[4:])
    ):

        intersection = area_intersections[area]
        union = area_unions[area]

        if union.item() == 0:
            raise ValueError(
                f"Validation IoU is undefined for {area}. "
                "Neither predictions nor ground-truth masks "
                "contain lava pixels."
            )

        area_ious[area] = (
            intersection / union
        ).item()

    if not area_ious:
        raise ValueError(
            "No validation areas were found."
        )

    # Give every geographical area equal weight.

    mean_area_iou = (
        sum(area_ious.values()) / len(area_ious)
    )

    return average_loss, overall_iou, area_ious, mean_area_iou



def main():
    #  ==========================================================

    #  MODEL OUTPUTS

    #  ==========================================================



    #  Main folder containing the outputs from all training runs.

    MODEL_OUTPUT_FOLDER = Path("model_outputs")

    MODEL_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)



    #  Create a unique timestamp for this training run.

    #

    #  This allows outputs from different experiments to be retained

    #  in separate folders.

    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")



    #  Create a separate folder for this training run.

    RUN_FOLDER = MODEL_OUTPUT_FOLDER / RUN_TIMESTAMP

    RUN_FOLDER.mkdir(parents=True, exist_ok=True)



    #  Spreadsheet containing the training settings and epoch history.

    RUN_XLSX_PATH = RUN_FOLDER / "training_run.xlsx"



    #  Permanent copy of the best model checkpoint from this

    #  particular training run.

    RUN_MODEL_PATH = RUN_FOLDER / "unet_lava_best.pth"




    #  ==========================================================

    #  CREATE TRAINING DATASET

    #  ==========================================================



    #  Load the tiled training images and masks.

    #

    #  Augmentation is controlled by USE_AUGMENTATION.

    train_dataset_original = LavaFlowDataset(
        images_dir="data/training/images",
        masks_dir="data/training/masks",
        texture_dir="data/training/texture",
        augment=False
    )

    train_dataset_augmented = LavaFlowDataset(
        images_dir="data/training/images",
        masks_dir="data/training/masks",
        texture_dir="data/training/texture",
        augment=USE_AUGMENTATION,
        brightness_augment=USE_BRIGHTNESS_AUGMENTATION
    )

    # Retain the original dataset variable for existing reporting and logging.
    train_dataset = train_dataset_original

    image, mask, filename = train_dataset[0]
    print(image.shape)
    print(mask.shape)
    print(mask.dtype)
    print(torch.unique(mask))

    # Both passes cover every training tile, in independently shuffled order.
    train_loader_original = DataLoader(
        train_dataset_original,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    train_loader_augmented = DataLoader(
        train_dataset_augmented,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    print(f"Training images: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader_original) + len(train_loader_augmented)}")
    print()


    #  ==========================================================

    #  CREATE VALIDATION DATASET

    #  ==========================================================



    #  Create the validation dataset.

    #

    #  This is kept separate from training so model performance can

    #  be assessed on images that are not used to update the weights.

    #

    #  Augmentation is not enabled for validation.

    val_dataset = LavaFlowDataset(

        images_dir="data/validation/images",

        masks_dir="data/validation/masks",
        texture_dir="data/validation/texture"

    )





    #  ==========================================================

    #  CREATE VALIDATION DATALOADER

    #  ==========================================================



    #  Create the validation DataLoader.

    #

    #  shuffle=False keeps validation deterministic.

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )



    print(f"Validation images: {len(val_dataset)}")

    print(f"Validation batches per epoch: {len(val_loader)}")

    # Identify validation areas from the dataset filenames.

    validation_areas = sorted(
        {
            get_validation_area(path.name)
            for path in val_dataset.image_files
        },
        key=lambda name: int(name[4:])
    )

    print(f"Validation areas: {validation_areas}")

    '''
    #  ==========================================================

    #  PRE-TRAINING CHECK

    #  ==========================================================



    #  Count the number of tiled testing images.

    #

    #  This is displayed for reference before training begins but

    #  the testing dataset is not used during model training.

    num_test_tiles = len(

        list(Path("data/testing/images").glob("*.png"))

    )



    #  Display the dataset sizes and current training configuration.

    print()

    print("=" * 60)

    print("PRE-TRAINING CHECK")

    print("=" * 60)



    print(f"Training tiles:       {len(train_dataset)}")

    print(f"Validation tiles:     {len(val_dataset)}")

    print(f"Testing tiles:        {num_test_tiles}")

    print()



    print(f"Batch size:           {BATCH_SIZE}")

    print(f"Number of epochs:     {NUM_EPOCHS}")

    print(f"Learning rate:        {LEARNING_RATE}")

    print(f"Positive weights:     {POS_WEIGHTS}")

    print(f"Augmentation:         {USE_AUGMENTATION}")

    print(f"Resume training:      {RESUME_TRAINING}")



    print("=" * 60)

    print()



    #  Ask whether the training dataset has changed since the previous run.

    #

    #  Changes to the dataset can alter the class distribution and may

    #  therefore require the class weights to be recalculated.

    dataset_changed = input(

        "Have you changed the training dataset since the previous training run? "

        "(y/n): "

    ).strip().lower()



    #  If the dataset has changed, explicitly require confirmation

    #  that its class distribution and weights have been checked.

    if dataset_changed == "y":

        weights_checked = input(

            "Have you checked that POS_WEIGHTS are appropriate "

            "for the current training dataset? "

            "(y/n): "

        ).strip().lower()



        #  Cancel training if the updated dataset has not yet been analysed.

        if weights_checked != "y":

            print()

            print(

                "Training cancelled. Check the foreground class "

                "distribution and POS_WEIGHTS before training again."

            )

            raise SystemExit



    #  Require a final confirmation after displaying the current settings.

    confirmation = input(

        "Have you checked the training settings above and are you ready "

        "to start training? (y/n): "

    ).strip().lower()



    if confirmation != "y":

        print("Training cancelled.")

        raise SystemExit



    '''

    #  ==========================================================

    #  CREATE EXCEL TRAINING LOG

    #  ==========================================================



    #  Create a new Excel workbook for this training run.

    workbook = Workbook()





    #  ----------------------------------------------------------

    #  RUN SUMMARY SHEET

    #  ----------------------------------------------------------



    #  Use the default worksheet as the run summary.

    summary_sheet = workbook.active

    summary_sheet.title = "Run Summary"



    #  Add the worksheet title.

    summary_sheet["A1"] = "U-Net Training Run"

    summary_sheet["A1"].font = Font(bold=True, size=14)



    #  Record the main settings and dataset information used for

    #  this training run.

    summary_sheet["A3"] = "Run timestamp"

    summary_sheet["B3"] = RUN_TIMESTAMP



    summary_sheet["A4"] = "Batch size"

    summary_sheet["B4"] = BATCH_SIZE



    summary_sheet["A5"] = "Number of epochs"

    summary_sheet["B5"] = NUM_EPOCHS



    summary_sheet["A6"] = "Learning rate"

    summary_sheet["B6"] = LEARNING_RATE



    summary_sheet["A7"] = "Number of output channels"

    summary_sheet["B7"] = 1



    summary_sheet["A8"] = "Tile size"

    summary_sheet["B8"] = f"{train_dataset[0][0].shape[1]} x {train_dataset[0][0].shape[2]}"



    summary_sheet["A9"] = "Training tiles"

    summary_sheet["B9"] = len(train_dataset)



    summary_sheet["A10"] = "Validation tiles"

    summary_sheet["B10"] = len(val_dataset)



    summary_sheet["A11"] = "Testing tiles"

    summary_sheet["B11"] = len(

        list(Path("data/testing/images").glob("*.png"))

    )



    summary_sheet["A12"] = "Training folder"

    summary_sheet["B12"] = "data/training/images"



    summary_sheet["A13"] = "Training masks folder"

    summary_sheet["B13"] = "data/training/masks"



    summary_sheet["A14"] = "Validation folder"

    summary_sheet["B14"] = "data/validation/images"



    summary_sheet["A15"] = "Validation masks folder"

    summary_sheet["B15"] = "data/validation/masks"



    summary_sheet["A16"] = "Loss function"

    summary_sheet["B16"] = "Binary Cross Entropy + Binary Dice"



    summary_sheet["A17"] = "Positive class weights"

    summary_sheet["B17"] = str(POS_WEIGHTS)



    summary_sheet["A18"] = "Geometric augmentation"

    summary_sheet["B18"] = str(USE_AUGMENTATION)



    summary_sheet["A19"] = "Optimiser"

    summary_sheet["B19"] = "Adam"



    summary_sheet["A20"] = "Scheduler"

    summary_sheet["B20"] = "ReduceLROnPlateau"



    summary_sheet["A21"] = "Model checkpoint"

    summary_sheet["B21"] = str(RUN_MODEL_PATH)



    #  Record the class-index definitions used by the model.

    summary_sheet["A23"] = "Output channel 0"
    summary_sheet["B23"] = "Lava"

    summary_sheet["A24"] = "Output channel 1"
    summary_sheet["B24"] = "Not used"

    summary_sheet["A25"] = "Background handling"
    summary_sheet["B25"] = "Lava probability below threshold"

    # Two-pass training and input-channel settings.
    summary_sheet["A26"] = "Input channels"
    summary_sheet["B26"] = "SAR (0-1) and texture (raw float32 ratios)"
    summary_sheet["A27"] = "Training passes per epoch"
    summary_sheet["B27"] = "Original, then augmented"
    summary_sheet["A28"] = "Brightness augmentation"
    summary_sheet["B28"] = str(USE_BRIGHTNESS_AUGMENTATION)
    summary_sheet["A31"] = "Brightness augmentation parameters"
    summary_sheet["B31"] = "SAR only; multiplier U(0.7, 1.3); probability 0.5"
    summary_sheet["B29"] = "data/training/texture"
    summary_sheet["A30"] = "Validation texture folder"
    summary_sheet["B30"] = "data/validation/texture"




    #  Make the summary readable.

    summary_sheet.column_dimensions["A"].width = 25

    summary_sheet.column_dimensions["B"].width = 60





    #  ----------------------------------------------------------

    #  EPOCH HISTORY SHEET

    #  ----------------------------------------------------------



    #  Create a second worksheet containing one row for every

    #  completed training epoch.

    history_sheet = workbook.create_sheet("Epoch History")



    #  Define the information recorded after each epoch.

    history_headers = [
        "Epoch",
        "Train Loss",
        "Validation Loss",
        "Lava IoU",
        "Learning Rate",
        "Best Checkpoint Score",
        "Checkpoint Saved",
        "Mean Area IoU",
        "Area IoU Std Dev",
        "Checkpoint Score",
    ]

    history_headers.extend(
        f"{area} IoU"
        for area in validation_areas
    )

    #  Add and format the column headings.

    for column, header in enumerate(history_headers, start=1):

        cell = history_sheet.cell(row=1, column=column)

        cell.value = header

        cell.font = Font(bold=True)

        cell.fill = PatternFill("solid", fgColor="D9EAF7")

        cell.alignment = Alignment(horizontal="center")



    #  Set sensible column widths.

    column_widths = {

        "A": 10,

        "B": 14,

        "C": 18,

        "D": 18,

        "E": 14,

        "F": 14,

        "G": 20,

        "H": 15,

        "I": 18,

        "J": 18,

    }



    for column, width in column_widths.items():

        history_sheet.column_dimensions[column].width = width


    from openpyxl.utils import get_column_letter

    for column in range(8, len(history_headers) + 1):

        column_letter = get_column_letter(column)

        history_sheet.column_dimensions[column_letter].width = 18


    #  ==========================================================

    #  SAVE INITIAL TRAINING LOG

    #  ==========================================================



    #  Save the workbook before training starts so that the run

    #  configuration is recorded immediately.

    workbook.save(RUN_XLSX_PATH)



    print(f"Training log will be saved to: {RUN_XLSX_PATH}")





    #  ==========================================================

    #  CREATE U-NET MODEL

    #  ==========================================================



    #  Use a CUDA-compatible GPU if one is available.

    #  Otherwise, train on the CPU.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")



    #  Create the U-Net and move its parameters to the selected device.

    model = UNet().to(device)





    #  ==========================================================

    #  CREATE LOSS FUNCTION

    #  ==========================================================



    #  Convert positive-class weights into a tensor.

    #

    #  Channel 0 = Faults

    #

    #  These weights control the contribution of positive pixels

    #  relative to negative pixels for each foreground class.



    pos_weights = torch.tensor(

        POS_WEIGHTS,

        dtype=torch.float32,

        device=device

    )



    #  Reshape weights for broadcasting across:

    #  [batch, channel, height, width]

    pos_weights = pos_weights.view(1, 1, 1, 1)



    #  Create binary cross-entropy loss.

    #

    #  Each output channel is treated as an independent

    #  binary segmentation problem.

    #

    #  Background pixels have target 0 in both channels.

    #

    #  BCEWithLogitsLoss applies sigmoid internally,

    #  so the model must supply raw logits.



    criterion_bce = nn.BCEWithLogitsLoss(

        pos_weight=pos_weights

    )



    #  ==========================================================

    #  CREATE OPTIMISER

    #  ==========================================================



    #  Adam updates the model parameters using gradients calculated

    #  during backpropagation.

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)





    #  ==========================================================

    #  CREATE LEARNING-RATE SCHEDULER

    #  ==========================================================



    #  ReduceLROnPlateau monitors lava IoU.

    #

    #  If the score does not improve for 10 epochs, the learning

    #  rate is multiplied by 0.5.

    #

    #  The learning rate cannot fall below 1e-6.

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="max",      #  Higher score is better

        factor=0.5,      #  Halve the learning rate

        patience=10,     #  Wait 10 epochs with no improvement

        min_lr=1e-6

    )





    #  ==========================================================

    #  INITIALISE BEST VALIDATION SCORE

    #  ==========================================================



    #  Track the best lava IoU seen so far.

    #

    #  This determines when a new best model checkpoint is saved.

    best_val_score = float("-inf")





    #  ==========================================================

    #  OPTIONALLY RESUME TRAINING

    #  ==========================================================



    #  Load a saved checkpoint if requested.

    #

    #  The checkpoint can contain both the model weights and the

    #  best validation score.

    if RESUME_TRAINING and os.path.exists(MODEL_PATH):



        checkpoint = torch.load(MODEL_PATH, map_location=device)



        #  Newer checkpoints store both the weights and the best score.

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:



            #  Restore the trained model parameters.

            model.load_state_dict(checkpoint["model_state_dict"])



            #  Restore the previous best validation score so that the

            #  checkpoint threshold is not reset when training resumes.

            if checkpoint.get("selection_metric") != "mean_minus_std":
                raise ValueError(
                    "Checkpoint uses an older selection metric. "
                    "Cannot resume with the new combined score."
                )

            best_val_score = float(checkpoint["best_val_score"])



            print(f"Loaded checkpoint from {MODEL_PATH}")

            print(f"Best validation score so far: {best_val_score:.6f}")



        #  Older checkpoints may contain only the model weights.

        #

        #  In that case, load the weights and evaluate the model on

        #  the current validation dataset to establish its score.

        else:

            model.load_state_dict(checkpoint)

            print(f"Loaded model weights from {MODEL_PATH}")



            #  Evaluate the loaded model using the binary loss.

            _, _, area_ious, mean_area_iou = evaluate(
                model,
                val_loader,
                criterion_bce,
                device
            )

            best_val_score = (
                mean_area_iou
                - IOU_STD_PENALTY * pstdev(area_ious.values())
            )







            print(f"Validation score for loaded model: {best_val_score:.6f}")



    #  If resume training was requested but no checkpoint exists,

    #  training continues with a newly initialised model.

    elif RESUME_TRAINING:

        print(f"No saved model found at {MODEL_PATH}, starting from scratch")





    #  ==========================================================

    #  TRAINING LOOP

    #  ==========================================================



    #  Train for the chosen number of epochs.

    for epoch in range(NUM_EPOCHS):





        #  ------------------------------------------------------

        #  TRAINING PHASE

        #  ------------------------------------------------------



        #  Put the model into training mode.

        model.train()



        #  Accumulate training loss across all batches in this epoch.

        running_train_loss = 0.0



        # Pass 1: original SAR and texture, no augmentation.
        # Pass 2: same paired tiles, with geometric augmentation and
        # a 50% chance of SAR-only brightness multiplication (0.7-1.3).
        for training_loader in (train_loader_original, train_loader_augmented):
            for images, masks, filenames in training_loader:
                images = images.to(device)
                masks = masks.to(device)
                predictions = model(images)
                bce_loss = criterion_bce(predictions, masks)
                d_loss = dice_loss_binary(predictions, masks)
                loss = bce_loss + d_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

        # Average across all batches in both passes.
        average_train_loss = running_train_loss / (
            len(train_loader_original) + len(train_loader_augmented)
        )


        #  ------------------------------------------------------

        #  ------------------------------------------------------

        #  VALIDATION PHASE

        #  ------------------------------------------------------


        average_val_loss, overall_val_iou, area_ious, mean_area_iou = evaluate(
            model,
            val_loader,
            criterion_bce,
            device
        )

        # Overall pixel-pooled lava IoU.

        lava_iou = overall_val_iou[0].item()

        if torch.isnan(overall_val_iou[0]):

            raise ValueError(
                "Overall validation IoU is undefined for lava."
            )

        # Use mean regional IoU for learning-rate scheduling.

        mean_foreground_iou = mean_area_iou

        # Calculate variation in IoU across validation areas.

        area_iou_std = pstdev(area_ious.values())

        # Combined score for checkpoint selection.

        checkpoint_score = (
            mean_area_iou - IOU_STD_PENALTY * area_iou_std
        )



        #  ------------------------------------------------------

        #  UPDATE LEARNING RATE

        #  ------------------------------------------------------



            #  Give the current lava IoU to the learning-rate
            #  scheduler.

        scheduler.step(mean_foreground_iou)





            #  ------------------------------------------------------

            #  SAVE BEST MODEL

            #  ------------------------------------------------------



            #  Save a new checkpoint whenever the current lava
            #  IoU exceeds the best score previously observed.

        if checkpoint_score > best_val_score:



                #  Update the stored best score.

            best_val_score = checkpoint_score



                #  Store the model weights, validation score, and run

                #  timestamp together in the checkpoint.

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "best_val_score": best_val_score,
                "selection_metric": "mean_minus_std",
                "iou_std_penalty": IOU_STD_PENALTY,
                "run_timestamp": RUN_TIMESTAMP,
            }



                #  Save to the normal model path used by evaluate.py.

                #

                #  This file represents the currently active best model.

            torch.save(

                checkpoint_data,

                MODEL_PATH

            )



                #  Also save a permanent copy specific to this training run.

                #

                #  This preserves the checkpoint inside its associated

                #  model_outputs timestamp folder.

            torch.save(

                checkpoint_data,

                RUN_MODEL_PATH

            )



            checkpoint_saved = "Yes"



            print(
                f"Saved new best model with combined score: "
                f"{best_val_score:.6f}"
            )


        else:

            checkpoint_saved = "No"





        #  ------------------------------------------------------

        #  RECORD THIS EPOCH IN THE EXCEL WORKBOOK

        #  ------------------------------------------------------



        #  Read the learning rate currently being used by the optimiser.

        current_lr = optimizer.param_groups[0]["lr"]



        #  Add the next epoch to the first unused worksheet row.

        history_row = history_sheet.max_row + 1



        #  Record the epoch metrics.

        history_sheet.cell(row=history_row, column=1, value=epoch + 1)
        history_sheet.cell(row=history_row, column=2, value=average_train_loss)
        history_sheet.cell(row=history_row, column=3, value=average_val_loss)
        history_sheet.cell(row=history_row, column=4, value=lava_iou)
        history_sheet.cell(row=history_row, column=5, value=current_lr)
        history_sheet.cell(row=history_row, column=6, value=best_val_score)
        history_sheet.cell(row=history_row, column=7, value=checkpoint_saved)
        history_sheet.cell(
            row=history_row,
            column=8,
            value=mean_area_iou
        )

        history_sheet.cell(
            row=history_row,
            column=9,
            value=area_iou_std
        )

        history_sheet.cell(
            row=history_row,
            column=10,
            value=checkpoint_score
        )

        for column, area in enumerate(validation_areas, start=11):

            history_sheet.cell(
                row=history_row,
                column=column,
                value=area_ious[area]
            )

        #  Format numerical metrics to six decimal places.

        for column in list(range(2, 7)) + list(range(8, len(history_headers) + 1)):

            history_sheet.cell(

                row=history_row,

                column=column

            ).number_format = "0.000000"

        #  Save after every epoch so the training history is retained

        #  throughout the run.

        workbook.save(RUN_XLSX_PATH)





        #  ------------------------------------------------------

        #  PRINT EPOCH RESULTS

        #  ------------------------------------------------------



        #  Read the current learning rate after the scheduler update.

        current_lr = optimizer.param_groups[0]["lr"]



        #  Display the results from the completed epoch.

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}\n"
            f"Train loss:        {average_train_loss:.6f}\n"
            f"Val loss:          {average_val_loss:.6f}\n"
            f"Overall lava IoU:  {lava_iou:.6f}\n"
            f"Mean area IoU:     {mean_area_iou:.6f}\n"
            f"Area IoU std dev:  {area_iou_std:.6f}\n"
            f"Checkpoint score:  {checkpoint_score:.6f}\n"
            f"Learning rate:     {current_lr:.2e}"
        )

        for area in validation_areas:

            print(
                f"  {area} lava IoU: {area_ious[area]:.6f}"
            )

        print()


if __name__ == "__main__":
    main()