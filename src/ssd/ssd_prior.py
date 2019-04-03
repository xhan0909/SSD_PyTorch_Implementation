import numpy as np
import torch
from ..transform.__boxtools__ import *


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance,
                 iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = center_bb(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.iscuda = torch.cuda.is_available()

    def __call__(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        boxes, labels = match_prior_with_truth(gt_boxes, gt_labels,
                                               self.corner_form_priors,
                                               self.iou_threshold)
        center_form_boxes = torch.empty(boxes.shape)
        for i, box in enumerate(boxes):
            center_form_boxes[i] = bb_center(box)
        if self.iscuda:
            center_form_boxes = center_form_boxes.cuda(non_blocking=True)
        locations = convert_boxes_to_locations(center_form_boxes.float(),
                                               self.center_form_priors,
                                               self.center_variance,
                                               self.size_variance)
        return locations, labels
