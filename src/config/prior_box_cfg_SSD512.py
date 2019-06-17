import collections
import numpy as np

image_size = 512
image_mean = np.array([123, 117, 104])
image_std = 1.0

num_classes = 2  # if more classes are added, change this number
batch_size = 32  # [TRAIN]
neg_pos_ratio = 2  # [TRAIN]

center_variance = 0.1  # [TRAIN]
size_variance = 0.2  # [TRAIN]
iou_threshold = 0.2  # [TEST]
prob_threshold = 0.55  # [TEST]

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
Spec = collections.namedtuple('Spec',
                              ['feature_map_size', 'shrinkage',
                               'box_sizes', 'aspect_ratios'])

# the SSD512 original specs, [TRAIN]
specs = [
    Spec(64, 8, SSDBoxSizes(36, 77), [2]),
    Spec(32, 16, SSDBoxSizes(77, 154), [2, 3]),
    Spec(16, 32, SSDBoxSizes(154, 230), [2, 3]),
    Spec(8, 64, SSDBoxSizes(230, 307), [2, 3]),
    Spec(4, 128, SSDBoxSizes(307, 384), [2, 3]),
    Spec(2, 256, SSDBoxSizes(384, 460), [2]),
    Spec(1, 512, SSDBoxSizes(460, 537), [2])
]
