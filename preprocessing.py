import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import patches, patheffects

from src.transform.__boxtools__ import *


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


def prepare_dataset(dict_anno, target_im_path):
    """
    Prepare data set.

    :param dict_anno: dictionary format annotation
    :return: A data frame with three columns: file path, original bboxes, resized bboxes
    """
    data_bbox_multi = {
        'fn': [target_im_path / f'{idx:06d}.jpg' for idx in dict_anno.keys()],
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


def resize_img_and_write(IMG_PATH, NEW_PATH, new_size):
    """Resize the original images and write the resized images
    to a new directory.

    :param IMG_PATH: original image path
    :param NEW_PATH: resized image path
    :param new_size: target image size
    :return: None
    """
    for path in tqdm(list(IMG_PATH.iterdir())):
        img = cv2.imread(str(path))
        x_resized = cv2.resize(img, (new_size, new_size))
        fname = str(path).split('/')[-1]
        cv2.imwrite(str(NEW_PATH/fname), x_resized)
    tqdm.write(f'Successfully resize images. New path: {NEW_PATH}')


def resize_bbox(orig_shape, bbox, target_size):
    """Resize bounding boxes accordingly.

    :param orig_shape: shape of original image
    :param bbox: bounding boxes
    :param target_size: size of the resized image
    :return: new boxes (String)
    """
    Y = make_bb_px(bbox, orig_shape)
    y_resize = cv2.resize(Y, (target_size, target_size))
    y = bb_hw_numpy(to_bb(y_resize))
    return ' '.join([str(int(n)) for n in y])


def resize_annotation_bbox(orig_im_path, annotation):
    """Generate new annotation data frame with resized image path
    and new bounding boxes' coordinates.

    :param orig_im_path: original image path
    :param annotation: original annotation data frame
    :return:
    """
    y_new = []
    for i, bbox in enumerate(annotation['bbox']):
        path = annotation['fn'][i]
        # the original microfiber images has different sizes
        img = cv2.imread(str(orig_im_path / path.split('/')[-1]))
        r, c, _ = img.shape  # original img size
        y = np.array([int(b) for b in bbox.split()])
        if len(y) == 4:
            y_new.append(resize_bbox((r, c), y))
        elif len(y) > 4:
            resized_bb = [resize_bbox((r, c), y[i * 4:(i * 4 + 4)]) for i in
                          range(int(len(y) / 4))]
            y_new.append(' '.join(resized_bb))

    annotation['bbox_300'] = y_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_im_path", help="original image path")
    parser.add_argument("resized_im_path", help="resized image path")
    parser.add_argument("json_path", help="json format annotation path")
    parser.add_argument("df_path", help="annotation data frame path")
    parser.add_argument("target_size", help="target image size after resizing")

    args = parser.parse_args()

    if not os.path.exists(args.resized_im_path):
        os.makedirs(args.resized_im_path)

    # resize images to target size
    print('Resizing images...')
    resize_img_and_write(args.orig_im_path, args.resized_im_path,
                         args.target_size)

    # make annotation dictionary with original image and bboxes
    print('Making annotations with resized images...')
    annotations = json.load(open(args.json_path/'train.json'))
    vals = json.load(open(args.json_path/'validation.json'))
    train_anno = make_anno_dict(annotations)
    train_bbox_multi_df = prepare_dataset(train_anno, args.resized_im_path)
    val_anno = make_anno_dict(vals)
    val_bbox_multi_df = prepare_dataset(val_anno, args.resized_im_path)
    print('Annotations done!')

    # resize bounding boxes accordingly
    print('Updating annotations with resized bounding boxes...')
    resize_annotation_bbox(train_bbox_multi_df)
    resize_annotation_bbox(val_bbox_multi_df)
    print('Annotations updated!')

    # write resized boxes to csv for future use
    print('Saving annotations to csv files...')
    train_bbox_multi_df.to_csv(args.df_path/'train_anno.csv', index=False)
    val_bbox_multi_df.to_csv(args.df_path/'val_anno.csv', index=False)
    print('Preprocessing completed. Exit...')
