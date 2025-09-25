import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np

verificar ='leaf/masks_png/train/leaf_001_002_segmentation.png'
verificar = Image.open(verificar).convert('L')
array_mask = np.array(verificar)
array = torch.from_numpy(array_mask).long()

print(np.unique(array))