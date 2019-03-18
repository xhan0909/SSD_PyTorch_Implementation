import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import patches, patheffects

from __boxtools__ import to_bb, hw_bb, bb_hw, make_bb_px


def make_anno_dict(anno_json):
    """
    Convert json annotation file to dictionary.

    :param anno_json: json format annotation
    :return: dictionary format annotation
    """
    anno_dict = defaultdict(list)
    for anno in anno_json['annotations']:
        if not anno['ignore']:
            bb = np.array(anno['bbox'])
            anno_dict[anno['image_id']].append(
                (anno['category_id'], bb))
    return anno_dict


def prepare_dataset(dict_anno):
    """
    Prepare data set.

    :param dict_anno: dictionary format annotation
    :return: A data frame with three columns: file path, original bboxes, resized bboxes
    """
    data_bbox_multi = {
        'fn': [NEW_PATH / f'{idx:06d}.jpg' for idx in dict_anno.keys()],
        'label': [list(zip(*dict_anno[idx]))[0] for idx in dict_anno.keys()],
        'bbox': [np.hstack([*list(zip(*dict_anno[idx]))[1]]) for idx in
                 dict_anno.keys()]}
    data_bbox_multi_df = pd.DataFrame.from_dict(data_bbox_multi,
                                                orient='columns')
    data_bbox_multi_df['label'] = data_bbox_multi_df['label'].apply(
        lambda x: ' '.join([str(n) for n in x]))
    data_bbox_multi_df['bbox'] = data_bbox_multi_df['bbox'].apply(
        lambda x: ' '.join([str(n) for n in x]))

    return data_bbox_multi_df


def resize_img_and_write(IMG_PATH, size, NEW_PATH):
    """
    Resize original images.

    :param IMG_PATH: file path of the original image (before resizing)
    :param size: target image size => (size, size)
    :param NEW_PATH: file path of the resized image
    """
    for path in tqdm(list(IMG_PATH.iterdir())):
        img = cv2.imread(str(path))
        x_resized = cv2.resize(img, (size, size))
        fname = str(path).split('/')[-1]
        cv2.imwrite(str(NEW_PATH/fname), x_resized)
    tqdm.write(f'Successfully resize images. New path: {NEW_PATH}')


def resize_bbox(orig_shape, bbox, target_shape=(300, 300)):
    """
    Resize bounding boxes.

    :param orig_shape: image shape before resizing
    :param bbox: bounding boxes
    :param target_shape: target image shape
    :return: coordinates of the resized bounding boxes (hw format)
    """
    Y = make_bb_px(bbox, orig_shape)
    y_resize = cv2.resize(Y, target_shape)
    y = bb_hw(to_bb(y_resize))
    return ' '.join([str(int(n)) for n in y])


def resize_annotation_bbox(annotation):
    """
    Add a new column of resized bounding boxes to the annotation data frame.

    :param annotation: annotation data frame (from csv file)
    """
    y_300 = []
    for i, bbox in tqdm(enumerate(annotation['bbox'])):
        path = annotation['fn'][i]
        img = cv2.imread(str(IMG_PATH / str(path).split('/')[-1]))
        r, c, _ = img.shape  # original img size
        y = np.array([int(b) for b in bbox.split()])
        if len(y) == 4:
            y_300.append(resize_bbox((r, c), y))
        elif len(y) > 4:
            resized_bb = [resize_bbox((r, c), y[i * 4:(i * 4 + 4)]) for i in
                          range(int(len(y) / 4))]
            y_300.append(' '.join(resized_bb))

    annotation['bbox_300'] = y_300


if __name__ == '__main__':
    IMG_PATH = Path('../../VOCdevkit/VOC2007/JPEGImages')
    NEW_PATH = Path('../../VOCdevkit/VOC2007/JPEGImages_300_300')
    JSON_PATH = Path('../../json')
    DATA_PATH = Path('../../VOCdevkit/VOC2007/tmp')

    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)

    # make annotation dictionary with original image and bboxes
    annotations = json.load(open(JSON_PATH / 'pascal_train2007.json'))
    vals = json.load(open(JSON_PATH / 'pascal_val2007.json'))
    train_anno = make_anno_dict(annotations)
    train_bbox_multi_df = prepare_dataset(train_anno)
    val_anno = make_anno_dict(vals)
    val_bbox_multi_df = prepare_dataset(val_anno)

    # resize images to target size
    resize_img_and_write(IMG_PATH, 300, NEW_PATH)

    # resize bounding boxes accordingly
    resize_annotation_bbox(train_bbox_multi_df)
    resize_annotation_bbox(val_bbox_multi_df)

    # write resized boxes to csv for future use
    train_bbox_multi_df.to_csv(DATA_PATH / 'train_bbox_multi.csv', index=False)
    val_bbox_multi_df.to_csv(DATA_PATH / 'val_bbox_multi.csv', index=False)
