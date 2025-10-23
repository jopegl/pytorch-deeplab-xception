import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class LeafSegmentation(Dataset):
    NUM_CLASSES = 3 

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        
        base_dir = os.path.join("leaf_data")  
        self.image_dir = os.path.join(base_dir, split,"images")
        self.mask_dir = os.path.join(base_dir, split,"masks")
        self.area_dir = os.path.join(base_dir, split,'area')

        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.areas = sorted(os.listdir(self.area_dir))

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        area_path = os.path.join(self.area_dir, self.areas[index])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = torch.from_numpy(np.array(mask)).long()
        area = Image.open(area_path).convert('L')
        area = torch.from_numpy(np.array(area)).float()

        area /= 255.0

        return image, mask, area
