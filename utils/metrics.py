import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def area_accuracy(self, pred_area, target_area):
    # pred_area e target_area: (B, 1, H, W)

        print("Min max: ", target_area.min(), target_area.max())

        # garante formato consistente
        if pred_area.dim() == 3:  # (B, H, W)
            pred_area = pred_area.unsqueeze(1)
        if target_area.dim() == 3:
            target_area = target_area.unsqueeze(1)

        # resize do mapa previsto para o tamanho do mapa ground-truth
        pred_area = F.interpolate(pred_area, size=target_area.shape[2:], mode='bilinear', align_corners=False)

        # binarização
        pred_bin = (pred_area >= 0.5).float()
        target_bin = (target_area >= 0.5).float()

        # flatten para comparar pixel a pixel
        pred_flat = pred_bin.view(-1)
        target_flat = target_bin.view(-1)

        correct = (pred_flat == target_flat).sum().item()
        total = target_flat.numel()
        return correct / total





