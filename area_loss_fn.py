import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights):
        if predictions.dim() == 4 and predictions.shape[1] > 1:
            predictions = predictions.mean(dim=1, keepdim=True) 
        if predictions.shape[-2:] != targets.shape[-2:]:
            predictions = F.interpolate(predictions, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        #perguntar sobre o produto de hadamaard, não estou entendendo como usar os pesos
        raw_loss = F.mse_loss(predictions, targets, reduction='none')
        weighted_loss = raw_loss * weights
        area_loss_sum = torch.sum(weighted_loss)
        weight_sum = torch.sum(weights)
        avg_loss = area_loss_sum/weight_sum
        return avg_loss
    