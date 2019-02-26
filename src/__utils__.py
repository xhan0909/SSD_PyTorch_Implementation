import torch
import numpy as np
import itertools
from collections import defaultdict


def extract_metadata(json_path: str):
    """Make dictionaries:
    bounding boxes: {img_id:[(cat_id, bbox), (cat_id, bbox), ...]}
    categories: {id: category_name}
    """
    anno_json = json.load(open(json_path))
    anno_dict = defaultdict(list)
    categories = {d['id']: d['name'] for d in anno_json['categories']}
    for anno in anno_json['annotations']:
        if not anno['ignore']:
            bb = np.array(anno['bbox'])
            anno_dict[anno['image_id']].append(
                (anno['category_id'], bb))
    return anno_dict, categories


def save_model(m, p):
    torch.save(m.state_dict(), p)


def load_model(m, p):
    checkpoint = torch.load(p)
    m.load_state_dict(checkpoint)
    del checkpoint
    torch.cuda.empty_cache()


def set_trainable_attr(m, b=True):
    for p in m.parameters():
        p.requires_grad = b


def unfreeze(model, l):
    top_model = model.top_model
    set_trainable_attr(top_model[l])


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
    return hw[:, 0] * hw[:, 1]


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
    overlap_left_top = torch.max(gt[:, :2], pred[:, :2])
    overlap_right_bottom = torch.min(gt[:, 2:], pred[:, 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area_gt = area_of(gt[:, :2], gt[:, 2:])
    area_pred = area_of(pred[:, :2], pred[:, 2:])
    return overlap_area / (area_gt + area_pred - overlap_area + epsilon)


def match_prior_with_truth(gt_boxes, gt_labels, priors, iou_threshold):
    """ Match the prior box with ground-truth bounding boxes.
    (1) for every ground-truth box:
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

    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.

    Source: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/box_utils.py
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index

    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 20  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]

    return boxes, labels
