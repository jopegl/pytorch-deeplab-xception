from PIL import Image
from modeling.deeplab import DeepLab
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

image_path = '/content/pytorch-deeplab-xception/leaf_004_001.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
image = transform(image).unsqueeze(0)  # Adiciona dimensão de batch
checkpoint_path = '/content/pytorch-deeplab-xception/run/leaf/deeplab-xception/model_best.pth.tar'

model = DeepLab(num_classes=3,
                        backbone='xception',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

with torch.inference_mode():
    seg_out, _, area_out = model(image)
    seg_out = torch.softmax(seg_out, dim=1)
    pred_mask = seg_out.argmax(dim=1).squeeze().cpu().numpy()
    leaf_mask = seg_out[:,1:2] + seg_out[:,2:3]  
    final_area_pred = leaf_mask * area_out

final_area_pred = final_area_pred.squeeze().cpu().numpy()
seg_out = seg_out.squeeze().cpu().numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
plt.title('Input Image')
plt.subplot(1, 3, 2)
plt.imshow(pred_mask, cmap='jet', alpha=0.5)
plt.title('Predicted Segmentation Mask')
plt.subplot(1, 3, 3)
plt.imshow(final_area_pred, cmap='hot', alpha=0.5)
plt.title('Predicted Area Map')
plt.show()

