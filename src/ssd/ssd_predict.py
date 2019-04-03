import numpy as np


def predict(im_shape, locations, confidences, priors, top_k=-1,
            score_threshold=.3, prob_threshold=.5, iou_threshold=.5,
            candidate_size=200, sigma=.5, nms_type='hard'):
    """ Convert (confidence, location) predictions to (box, class, prob)
    as the final prediction that can be used to calculate metrics.

    :param im_shape: image shape
    :param locations: location predictions out of the model
    :param confidences: confidence predictions out of the model
    :param priors: prior boxes
    :param top_k: the minimum number of boxes to keep
    :param score_threshold: soft_nms, boxes with scores less than value are not considered.
    :param prob_threshold: boxes with probs less than this value are not considered.
    :param iou_threshold: IoU threshold (drop larger ious)
    :param candidate_size: top n candidates to look at
    :param sigma: soft_nms, sigma for the Gaussian penalty
    :param nms_type: 'hard' or 'soft'

    :return: (box, class, prob)
    """
    # this version of hard nm prediction is faster on CPU
    cpu_device = torch.device("cpu")
    locations = locations.to(cpu_device)
    confidences = confidences.to(cpu_device)

    # convert locations to boxes(hw)
    bb_real = __boxtools__.convert_locations_to_boxes(
        __boxtools__.bb_center(locations),
        priors,
        center_variance, size_variance)
    boxes, _ = __boxtools__.to_absolute_coords(im_shape,
                                               boxes=bb_real,
                                               labels=None)
    boxes = __boxtools__.center_hw(boxes.squeeze()).float()

    # Apply softmax on confidence to make [0, 1] probabilities
    confidences = F.softmax(confidences, dim=-1)

    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.size(1)):  # no background (0)
        probs = confidences[..., class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = __boxtools__.nms(box_probs,
                                     iou_threshold=iou_threshold,
                                     score_threshold=score_threshold,
                                     top_k=top_k,
                                     candidate_size=candidate_size,
                                     sigma=sigma,
                                     nms_type=nms_type)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))

    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    picked_box_probs = torch.cat(picked_box_probs)

    return (picked_box_probs[:, :4], torch.tensor(picked_labels),
            picked_box_probs[:, -1])
