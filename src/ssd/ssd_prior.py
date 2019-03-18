import numpy as np
import torch
from ..transform.__boxtools__ import *


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance,
                 iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = np.array([center_bb(prior) for prior in center_form_priors])
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        # check type: torch.Tensor
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.from_numpy(gt_boxes)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = match_prior_with_truth(gt_boxes, gt_labels,
                                               self.corner_form_priors,
                                               self.iou_threshold)
        center_form_boxes = []
        for box in boxes:
            center_form_boxes.append(bb_center(box))
        center_form_boxes = torch.from_numpy(np.array(center_form_boxes))
        locations = convert_boxes_to_locations(center_form_boxes,
                                               self.center_form_priors,
                                               self.center_variance,
                                               self.size_variance)
        return locations, labels
