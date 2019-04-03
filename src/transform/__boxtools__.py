import torch
import math
import numpy as np
import itertools
from typing import *
from collections import defaultdict


# Bounding boxes
def make_bb_px(y, x_shape):
    """Makes an image of size x rectangular bounding box"""
    Y = np.zeros(x_shape)
    y = hw_bb(y).astype(np.int)
    Y[y[0]:y[2], y[1]:y[3]] = 1.
    return Y


def to_bb(Y):
    """
    Convert mask Y to a corner format bounding box
    Assumes 0 as background nonzero object
    """
    cols, rows = np.nonzero(Y)
    if len(cols) == 0: return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row],
                    dtype=np.float32)


def hw_bb(bb: torch.Tensor) -> torch.Tensor:
    """Transform from width-height format to corner format.

    width-height: [X, Y, width, height]
    corner format: [Y_0, X_0, Y_right_bottom, X_right_bottom]
    """
    return torch.stack([bb[..., 1], bb[..., 0],
                      bb[..., 3] + bb[..., 1] - 1,
                      bb[..., 2] + bb[..., 0] - 1], dim=-1)


def bb_hw(bb: torch.Tensor) -> torch.Tensor:
    """Transform from corner format to width-height format.

    width-height: [X, Y, width, height]
    corner format: [Y_0, X_0, Y_right_bottom, X_right_bottom]
    """
    return torch.stack([bb[..., 1], bb[..., 0],
                      bb[..., 3] - bb[..., 1] + 1,
                      bb[..., 2] - bb[..., 0] + 1], dim=-1)


def bb_center(bb: torch.Tensor) -> torch.Tensor:
    """Given corner format [Y_0, X_0, Y_right_bottom, X_right_bottom]
    return cx-cy-height-width format."""
    w, h = bb[..., 3]-bb[..., 1], bb[..., 2]-bb[..., 0]
    cx = bb[..., 1] + w/2
    cy = bb[..., 0] + h/2
    if len(bb.shape) > 1:
        return torch.stack([cx, cy,h, w], dim=-1)
    return torch.Tensor([cx, cy, h, w])


def center_bb(bb: torch.Tensor) -> torch.Tensor:
    """Given cx-cy-height-width format
    return corner format [Y_0, X_0, Y_right_bottom, X_right_bottom]."""
    w, h = bb[..., 3], bb[..., 2]
    x = bb[..., 0] - w/2
    y = bb[..., 1] - h/2
    lb, rt = y+h, x+w
    if len(bb.shape) > 1:
        return torch.stack([y, x, lb, rt], dim=-1)
    # print(y.shape, x.shape, lb.shape, rt.shape)
    return torch.Tensor([y, x, lb, rt])


def center_hw(bb: torch.Tensor) -> torch.Tensor:
    """Given cx-cy-height-width format return x-y-width-height format."""
    w, h = bb[..., 3], bb[..., 2]
    x = bb[..., 0] - w/2
    y = bb[..., 1] - h/2
    if len(bb.shape) > 1:
        return torch.stack([x, y, w, h], dim=-1)
    return torch.Tensor([x, y, w, h])


def hw_center(bb: torch.Tensor) -> torch.Tensor:
    """Given x-y-width-height format return cx-cy-height-width format."""
    w, h = bb[..., 2], bb[..., 3]
    cx = bb[..., 0] + w/2
    cy = bb[..., 1] + h/2
    if len(bb.shape) > 1:
        return torch.stack([cx, cy, h, w], dim=-1)
    return torch.Tensor([cx, cy, h, w])


def to_absolute_coords(image_shape, boxes=None, labels=None):
    height, width = image_shape
    boxes[..., 0] *= width
    boxes[..., 2] *= width
    boxes[..., 1] *= height
    boxes[..., 3] *= height

    return boxes, labels


def to_percent_coords(image_shape, boxes=None, labels=None):
    height, width = image_shape
    boxes[..., 0] /= width
    boxes[..., 2] /= width
    boxes[..., 1] /= height
    boxes[..., 3] /= height

    return boxes, labels


# Prior boxes
def generate_ssd_priors(specs, image_size=300, clip=True):
    """
    Generate center format SSD Prior Boxes.

    :param specs: Specs about the shapes of sizes of prior boxes. i.e.
                    specs = [
                        Spec(38, 8, SSDBoxSizes(30, 60), [2]),
                        Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                        Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                        Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                        Spec(3, 100, SSDBoxSizes(213, 264), [2]),
                        Spec(1, 300, SSDBoxSizes(264, 315), [2])
                    ]
    :param image_size: image size.
    :param clip: boolean

    :return: a list of priors: [[center_x, center_y, h, w]].
             All the values are relative to the image size (300x300).

    Source: https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad
    """
    boxes = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            # small sized square box
            size = spec.box_sizes.min
            h = w = size / image_size
            boxes.append([
                x_center,
                y_center,
                h,
                w
            ])

            # big sized square box
            size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            boxes.append([
                x_center,
                y_center,
                h,
                w
            ])

            # change h/w ratio of the small sized box
            # based on the SSD implementation, it only applies ratio to the smallest size.
            # it looks weird.
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = np.sqrt(ratio)
                boxes.append([
                    x_center,
                    y_center,
                    h * ratio,
                    w / ratio
                ])
                boxes.append([
                    x_center,
                    y_center,
                    h / ratio,
                    w * ratio
                ])
    boxes = np.array(boxes)
    if clip:
        boxes = np.clip(boxes, 0.0, 1.0)

    return boxes


def area_of(left_top, right_bottom) -> torch.Tensor:
    """
    Compute the areas of rectangles given two corners.

    :param left_top: left top corner (x, y)
    :param right_bottom: right bottom corner (x+width, y+height)
    :return: the area
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(gt, pred):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    :param gt: ground truth boxes [Y_0, X_0, Y_right_bottom, X_right_bottom]
    :param pred: predicted boxes [Y_0, X_0, Y_right_bottom, X_right_bottom]
    :param epsilon: a small number to avoid 0 as denominator.
    :return: IoU values.
    """
    epsilon = 1e-10
    overlap_left_top = torch.max(gt[..., :2], pred[..., :2])
    overlap_right_bottom = torch.min(gt[..., 2:], pred[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area_gt = area_of(gt[..., :2], gt[..., 2:])
    area_pred = area_of(pred[..., :2], pred[..., 2:])
    return overlap_area / (area_gt + area_pred - overlap_area + epsilon)


def match_prior_with_truth(gt_boxes, gt_labels, priors, iou_threshold=0.5):
    """
    Match the prior box with ground-truth bounding boxes.
    (1) for every ground-truth box (target):
            match the ground-truth box with prior having the biggest IoU
    (2) for every prior:
            ious = IoU(prior, ground_truth_boxes)
            max_iou = max(ious)
            if max_iou > threshold:
                i = argmax(ious)
                match the prior with ground_truth_boxes[i]

    :param gt_boxes: ground truth boxes, [Y_0, X_0, Y_right_bottom, X_right_bottom]
    :param gt_labels: labels of targets, [Y_0, X_0, Y_right_bottom, X_right_bottom]
    :param priors: corner format prior boxes, [Y_0, X_0, Y_right_bottom, X_right_bottom]
    :param iou_threshold: IoU threshold, default 0.5
    :return:
        boxes (num_priors, 4): real values for priors, corner format
        labels (num_priros): labels for priors.

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    # if not isinstance(priors, torch.Tensor):
    #     priors = torch.from_numpy(priors).cuda()

    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), priors.unsqueeze(1))

    # (1) size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    # (2) size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index

    # 1 is used to make sure every target has a prior assigned
    # tensor.index_fill_(dim, index, val)
    best_target_per_prior.index_fill_(0, best_prior_per_target_index.squeeze(), 1.)

    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]

    return boxes, labels


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """
    Convert regression prediction (cx, cy, h, w) to real boxes
    based on prior boxes.

    Conversion formula:
        $$predicted\_center * center_variance
            = \frac{real\_center - prior\_center}{prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac{real\_hw}{prior\_hw}$$

    :param locations: output of SSD regression [batch_size, n_priors, 4]
                      box is in center form (cx, cy, h, w)
    :param priors: prior boxes [batch_size, n_priors, 4]
                   box is in center form (cx, cy, h, w)
    :param center_variance: float
    :param size_variance: float
    :return: real boxes [cx, cy, width, height]
    """
    # if not isinstance(locations, torch.Tensor):
    #     locations = torch.from_numpy(locations).float().cuda()
    # if not isinstance(priors, torch.Tensor):
    #     priors = torch.from_numpy(priors).float().cuda()

    if priors.dim() == locations.dim() - 1:
        priors = priors.unsqueeze(0)

    pred_center, pred_hw = locations[..., :2], locations[..., 2:]
    prior_center, prior_hw = priors[..., :2], priors[..., 2:]
    real_center = pred_center * center_variance * prior_center + prior_center
    real_hw = torch.exp(pred_hw * size_variance) * prior_hw

    return torch.cat((real_center, real_hw), dim=1)


def convert_boxes_to_locations(center_form_boxes, priors,
                               center_variance, size_variance):
    """
    Convert real boxes to locations (cx, cy, h, w) based on prior boxes.

    Conversion formula:
        $$predicted\_center * center_variance
            = \frac{real\_center - prior\_center}{prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac{real\_hw}{prior\_hw}$$

    :param center_form_boxes: real boxes [batch_size, n_priors, 4]
                              box is in center form (cx, cy, h, w)
    :param priors: prior boxes [batch_size, n_priors, 4]
                   box is in center form (cx, cy, h, w)
    :param center_variance: float
    :param size_variance: float
    :return: locations [cx, cy, width, height]
    """
    # if not isinstance(priors, torch.Tensor):
    #     priors = torch.from_numpy(priors).cuda(non_blocking=True)
    if priors.dim() == center_form_boxes.dim() - 1:
        priors = priors.unsqueeze(0)

    real_center, real_hw = center_form_boxes[..., :2], center_form_boxes[..., 2:]
    prior_center, prior_hw = priors[..., :2], priors[..., 2:]
    pred_center = (real_center - prior_center) / prior_hw / center_variance
    pred_hw = torch.log(real_hw / prior_hw) / size_variance

    return torch.cat((pred_center, pred_hw), dim=1)


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    :param loss: the loss for each example.
    :param labels: the labels.
    :param neg_pos_ratio: the ratio between the negative examples and positive examples.
    :return: boolean mask

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def nms(box_scores, iou_threshold=.5, score_threshold=.3,
        top_k=-1, candidate_size=200, sigma=0.5, nms_type='hard'):
    if nms_type == 'hard':
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size)
    if nms_type == 'soft':
        return soft_nms(box_scores, score_threshold, sigma, top_k)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Non-Maximum Suppression.

    :param box_scores: boxes in corner-form and probabilities.
    :param iou_threshold: intersection over union threshold.
    :param top_k: keep top_k results. If k <= 0, keep all the results.
    :param candidate_size: only consider the candidates with the highest scores.
    :return: a list of indexes of the kept boxes

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    probs = box_scores[:, -1] # predicted probability of each box
    boxes = box_scores[:, :4] # box location
    boxes_to_keep = []
    _, indexes = probs.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        boxes_to_keep.append(current.item())
        if 0 < top_k == len(boxes_to_keep) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0)
        )
        # pruning
        indexes = indexes[iou <= iou_threshold]

    return box_scores[boxes_to_keep, :]


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS python implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
        https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py

    :param box_scores: (N, 5), boxes in corner-form and probabilities.
    :param score_threshold: boxes with scores less than value are not considered.
    :param sigma: the parameter in score re-computation.
                  scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
    :param top_k: keep top_k results. If k <= 0, keep all the results.

    :return picked_box_scores: (K, 5), results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        # the first step is the same as hard_nms
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.Tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if 0 < top_k == len(picked_box_scores) or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])

        # update the pruning step with a Gaussian penalty function
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]

    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.Tensor([])
