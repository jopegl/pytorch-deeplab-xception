import numpy as np

arr = np.fromfile('pytorch-deeplab-xception/split_dataset/split_dataset/train/010/segmentation/leaf_010_001_segmentation.raw', dtype=np.int8)
print(arr.shape)