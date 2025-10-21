import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, weights):
        raw_loss = F.mse_loss(predictions, targets, reduction='none')
        weighted_loss = raw_loss * weights
        area_loss_sum = torch.sum(weighted_loss)
        weight_sum = torch.sum(weights)
        avg_loss = area_loss_sum/weight_sum
        return avg_loss
    