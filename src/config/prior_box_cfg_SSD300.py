import collections
import numpy as np

image_size = 300
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

# the SSD300 original specs, [TRAIN]
specs = [
    Spec(38, 8, SSDBoxSizes(30, 60), [2]),
    Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    Spec(3, 100, SSDBoxSizes(213, 264), [2]),
    Spec(1, 300, SSDBoxSizes(264, 315), [2])
]
