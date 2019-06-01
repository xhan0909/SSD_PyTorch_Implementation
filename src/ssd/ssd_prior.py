import torch
import numpy as np
from ..transform.__boxtools__ import *
from src.config.cuda_cfg import device


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance,
                 iou_threshold):
        self.center_form_priors = center_form_priors
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        center_form_boxes, labels = match_prior_with_truth(gt_boxes, gt_labels,
                                                           self.center_form_priors,
                                                           self.iou_threshold)
        # center_form_boxes = torch.empty(boxes.shape)
        # for i, box in enumerate(boxes):
        #     center_form_boxes[i] = bb_center(box)
        center_form_boxes = center_form_boxes.to(device, non_blocking=True)
        locations = convert_boxes_to_locations(center_form_boxes.float(),
                                               self.center_form_priors,
                                               self.center_variance,
                                               self.size_variance)
        return locations, labels
