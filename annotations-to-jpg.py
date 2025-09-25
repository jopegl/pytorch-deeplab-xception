import os
import numpy as np
from PIL import Image


masks_path = 'leaf/masks'
output_folder = 'leaf/masks_png'

width, height= 512, 512

for filename in os.listdir(masks_path):
  if filename.endswith('.raw'):
    path_in = os.path.join(masks_path, filename)
    data = np.fromfile(path_in, dtype=np.uint8)
    data = data.reshape((width, height))
    img = Image.fromarray(data)
    filename_out = filename.replace('raw','png')
    img.save(os.path.join(output_folder, filename_out))