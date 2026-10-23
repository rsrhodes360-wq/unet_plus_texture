

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Convert logits to probabilities.
        probabilities = F.softmax(predictions, dim=1)

        num_classes = predictions.shape[1]

        # Convert class labels to one-hot encoding.
        targets = F.one_hot(targets, num_classes=num_classes)
        targets = targets.permute(0, 3, 1, 2).float()

        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (
            union + self.smooth
        )

        # Average over all classes and all images.
        return 1.0 - dice.mean()