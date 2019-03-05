import torch
import numpy as np
import itertools
from collections import defaultdict


# Bounding boxes
def make_bb_px(y, x):
    """Makes an image of size x rectangular bounding box"""
    r, c, *_ = x.shape
    Y = np.zeros((r, c))
    y = hw_bb(y).astype(np.int)
    Y[y[0]:y[2], y[1]:y[3]] = 1.
    return Y


def to_bb(Y):
    """
    Convert mask Y to a bounding box
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


def hw_bb(bb):
    """Transform from width-height format to bounding box format.

    width-height: [X, Y, width, height]
    bounding-box: [Y, X, left-bottom, right-top]
    """
    return np.array([bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1])


def bb_hw(a):
    """Transform from bounding box format to width-height format.

    width-height: [X, Y, width, height]
    bounding-box: [Y, X, left-bottom, right-top]
    """
    return np.array([a[1], a[0], a[3] - a[1] + 1, a[2] - a[0] + 1])


def hw_center(bb):
    """Given x-y-width-height format return cx-cy-height-width format."""
    w, h = bb[2], bb[3]
    cx = bb[0] + w/2
    cy = bb[1] + h/2
    return np.array([cx, cy, h, w])


def center_hw(bb):
    """Given cx-cy-height-width format return x-y-width-height format."""
    w, h = bb[3], bb[2]
    x = bb[0] - w/2
    y = bb[1] - h/2
    return np.array([x, y, w, h])


def to_absolute_coords(image, boxes=None, labels=None):
    height, width, channels = image.shape
    boxes[:, 0] *= width
    boxes[:, 2] *= width
    boxes[:, 1] *= height
    boxes[:, 3] *= height

    return image, boxes, labels


def to_percent_coords(image, boxes=None, labels=None):
    height, width, channels = image.shape
    boxes[:, 0] /= width
    boxes[:, 2] /= width
    boxes[:, 1] /= height
    boxes[:, 3] /= height

    return image, boxes, labels


# Prior boxes
def generate_ssd_priors(specs, image_size=300, clip=True):
    """Generate SSD Prior Boxes.

    Args:
        specs: Specs about the shapes of sizes of prior boxes. i.e.
            specs = [
                Spec(38, 8, SSDBoxSizes(30, 60), [2]),
                Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                Spec(3, 100, SSDBoxSizes(213, 264), [2]),
                Spec(1, 300, SSDBoxSizes(264, 315), [2])
            ]
        image_size: image size.

    Returns:
        priors: a list of priors: [[center_x, center_y, h, w]]. All the values
                are relative to the image size (300x300).

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
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner (x, y).
        right_bottom (N, 2): right bottom corner (x+width, y+height).
    Returns:
        area (N): return the area.

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[:, :, 0] * hw[:, :, 1]


def iou_of(gt, pred, epsilon=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        gt (N, 4): ground truth boxes.
        pred (N or 1, 4): predicted boxes.
        epsilon: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    overlap_left_top = torch.max(gt[:, :, :2], pred[:, :, :2])
    overlap_right_bottom = torch.min(gt[:, :, 2:], pred[:, :, 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area_gt = area_of(gt[:, :, :2], gt[:, :, 2:])
    area_pred = area_of(pred[:, :, :2], pred[:, :, 2:])
    return overlap_area / (area_gt + area_pred - overlap_area + epsilon)


def match_prior_with_truth(gt_boxes, gt_labels, priors, iou_threshold):
    """ Match the prior box with ground-truth bounding boxes.
    (1) for every ground-truth box (target):
            match the ground-truth box with prior having the biggest IoU
    (2) for every prior:
            ious = IoU(prior, ground_truth_boxes)
            max_iou = max(ious)
            if max_iou > threshold:
                i = argmax(ious)
                match the prior with ground_truth_boxes[i]
    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form, (left_top, right_bottom)
        iou_threshold (float): IoU threshold

    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    if not isinstance(priors, torch.Tensor):
        priors = torch.from_numpy(priors)
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), priors.unsqueeze(1))

    # (1) size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    # (2) size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index

    # 2 is used to make sure every target has a prior assigned
    # tensor.index_fill_(dim, index, val)
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the background id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regression prediction (cx, cy, h, w) to real boxes
    based on prior boxes.

    Conversion formula:
        $$predicted\_center * center_variance
            = \frac{real\_center - prior\_center}{prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac{real\_hw}{prior\_hw}$$

    Args:
        locations: output of SSD regression [batch_size, n_priors, 4]
                   box is in center form (cx, cy, h, w)
        priors: prior boxes [batch_size, n_priors, 4]
                box is in center form (cx, cy, h, w)
        center_variance: float
        size_variance: float
    Returns:
        real boxes
    """
    if not isinstance(priors, torch.Tensor):
        priors = torch.from_numpy(priors)
    if priors.dim() == locations.dim() - 1:
        priors = priors.unsqueeze(0)
    pred_center, pred_hw = locations[:, :2], locations[:, 2:]
    prior_center, prior_hw = priors[:, :2], priors[:, 2:]
    real_center = pred_center * center_variance * prior_center + prior_center
    real_hw = torch.exp(pred_hw * size_variance) * prior_hw

    return torch.cat((real_center, real_hw), dim=1)


def convert_boxes_to_locations(center_form_boxes, priors,
                               center_variance, size_variance):
    """Convert real boxes to locations (cx, cy, h, w) based on prior boxes.

    Conversion formula:
        $$predicted\_center * center_variance
            = \frac{real\_center - prior\_center}{prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac{real\_hw}{prior\_hw}$$

    Args:
        center_form_boxes: real boxes [batch_size, n_priors, 4]
                           box is in center form (cx, cy, h, w)
        priors: prior boxes [batch_size, n_priors, 4]
                            box is in center form (cx, cy, h, w)
        center_variance: float
        size_variance: float

    Returns:
        locations
    """
    if not isinstance(priors, torch.Tensor):
        priors = torch.from_numpy(priors)
    if priors.dim() == center_form_boxes.dim() - 1:
        priors = priors.unsqueeze(0)
    real_center, real_hw = center_form_boxes[..., :2], center_form_boxes[..., 2:]
    prior_center, prior_hw = priors[..., :2], priors[..., 2:]
    pred_center = (real_center - prior_center) / prior_hw / center_variance
    pred_hw = torch.log(real_hw / prior_hw) / size_variance

    return torch.cat((pred_center, pred_hw), dim=1)


def hard_nm(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         boxes_to_keep: a list of indexes of the kept boxes

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    boxes_to_keep = []
    _, indexes = scores.sort(descending=True)
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
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[boxes_to_keep, :]
