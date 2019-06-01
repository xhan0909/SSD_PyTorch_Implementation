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
from src.transform.__imgtools__ import resize_bbox, resize_img_and_write


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
            fp = [item['file_name'] for item in anno_json['images']
                  if item['id'] == anno['image_id']]
            anno_dict[anno['image_id']].append(
                (anno['category_id'], bb, fp))
    return anno_dict


def prepare_dataset(dict_anno, target_im_path):
    """
    Prepare data set.

    :param dict_anno: dictionary format annotation
    :return: A data frame with three columns: file path, original bboxes, resized bboxes
    """
    target_im_path = Path(target_im_path)
    data_bbox_multi = {
        'fn_orig': [dict_anno[idx][0][2][0] for idx in dict_anno.keys()],
        'fn_new': [target_im_path/f'{idx}.jpeg' for idx in dict_anno.keys()],
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


def resize_annotation_bbox(df, target_size):
    """Generate new annotation data frame with resized image path
    and new bounding boxes' coordinates.

    :param df: annotation data frame
    """
    target_size = int(target_size)
    y_new = []
    for i, bbox in enumerate(df['bbox']):
        path = df['fn_orig'][i]
        # in case the original images may have different sizes
        img = cv2.imread(str(path))
        r, c, _ = img.shape  # original img size
        y = np.array([int(b) for b in bbox.split()])
        if len(y) == 4:
            y_new.append(resize_bbox((r, c), y, (target_size, target_size)))
        elif len(y) > 4:
            for j in range(int(len(y) / 4)):
                resized_bb = resize_bbox((r, c), y[j * 4:(j * 4 + 4)],
                                         (target_size, target_size))
                resized_bb = ' '.join([str(int(n)) for n in resized_bb])
            y_new.append(' '.join(resized_bb))

    df['bbox_300'] = y_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_im_path", help="text file containing original image path")
    parser.add_argument("resized_im_path", help="resized image path")
    parser.add_argument("json_path", help="json format annotation path")
    parser.add_argument("df_path", help="annotation data frame path")
    parser.add_argument("target_size", help="target image size after resizing")

    args = parser.parse_args()

    # if not os.path.exists(args.resized_im_path):
    #     os.makedirs(args.resized_im_path)

    # resize images to target size
    print('Resizing images...')
    resize_img_and_write(args.orig_im_path, args.resized_im_path,
                         args.target_size)

    # make annotation dictionary with original image and bboxes
    print('Making annotations with resized images...')
    annotations = json.load(open(Path(args.json_path)/'train.json'))
    vals = json.load(open(Path(args.json_path)/'valid.json'))
    train_anno = make_anno_dict(annotations)
    train_bbox_multi_df = prepare_dataset(train_anno, args.resized_im_path)
    val_anno = make_anno_dict(vals)
    val_bbox_multi_df = prepare_dataset(val_anno, args.resized_im_path)
    print('Annotations done!')

    # resize bounding boxes accordingly
    print('Updating annotations with resized bounding boxes...')
    resize_annotation_bbox(train_bbox_multi_df, args.target_size)
    resize_annotation_bbox(val_bbox_multi_df, args.target_size)
    print('Annotations updated!')

    # write resized boxes to csv for future use
    print('Saving annotations to csv files...')
    train_bbox_multi_df.to_csv(Path(args.df_path)/'train_anno.csv', index=False)
    val_bbox_multi_df.to_csv(Path(args.df_path)/'val_anno.csv', index=False)
    print('Preprocessing completed. Exit...')
