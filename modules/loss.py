import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Combination of BCEWithLogitsLoss and Dice Loss.
    """
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice
        
        # Combined Loss
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
