import numpy as np
from ..transform import __boxtools__


def compute_average_precision_per_class(num_true_cases, gt_boxes,
                                        prediction_file, iou_threshold=.5,
                                        use_2007_metric=False):
    """Compute average precision (AP) for each class.

    Reference:
    https://github.com/qfgaohao/pytorch-ssd/blob/master/eval_ssd.py

    :param num_true_cases: total number of labels of this class
    :param gt_boxes: Dict, {image_id: ground-truth boxes (corner-form?)}
    :param prediction_file: txt file, each row is "image_id score box"
    :param iou_threshold: IoU threshold, default 0.5
    :param use_2007_metric: whether to use PASCAL VOC2007 rule, default False

    :return: average precision for each class
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            # box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = __boxtools__.iou_of(gt_box, box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if (image_id, max_arg) not in matched:
                    true_positive[i] = 1
                    matched.add((image_id, max_arg))
                else:
                    false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    if use_2007_metric:
        return compute_voc2007_average_precision(precision, recall)
    else:
        return compute_average_precision(precision, recall)


def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition.
    It computes the under curve area of precision and recall. Recall follows the
    normal definition. Precision is a variant.

    pascal_precision[i] = typical_precision[i:].max()

    Reference: https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/measurements.py
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision)-1, 0, -1):
        precision[i-1] = np.maximum(precision[i-1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points+1] - recall[changing_points]) * precision[
        changing_points+1]
    return areas.sum()


def compute_voc2007_average_precision(precision, recall):
    """Compute mAP under PASCAL VOC2007 rule.

    Reference:
    https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/utils/measurements.py

    :param precision: precision score
    :param recall: recall score

    :return: average precision
    """
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.
    return ap
