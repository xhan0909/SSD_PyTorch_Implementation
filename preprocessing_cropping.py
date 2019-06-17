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
from src.transform.__imgtools__ import load_image, resize_bbox, crop_imgs, zero_padding


def get_all_images(path):
    img_list = []
    # p = Path('/data/xhan/Dropbox/USF MantaRay Data Share/Image Training Data')
    p = Path(path)
    for fname in p.glob('*/Microfibers/*.jp*'):  # we have both .jpg and .jpeg
        if 'Fall Testing Samples' not in str(fname):
            img_list.append(str(fname))

    return img_list


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


def crop_and_pad(df, t_size):
    df_new = df.copy()
    index_to_del = []

    for idx, row in tqdm(df.iterrows()):
        img = cv2.imread(str(row['fn_orig']))
        boxes = [int(c) for c in row['bbox'].split()]
        labels = [int(lab) for lab in row['label'].split()]
        r, c, _ = img.shape

        # for images smaller than 512 x 512: zero padding
        if r < t_size or c < t_size:
            x_resized = zero_padding(img, t_size)
            cv2.imwrite(str(row['fn_new']), x_resized)
        # for larger images: cropping
        else:
            index_to_del.append(idx)  # we delete the orig record after cropping
            for i in range(int(len(boxes) / 4)):
                bb = np.array(boxes[i*4:(i*4+4)])  # hw
                label = labels[i]
                rescaled_images = crop_imgs(img, y=bb, size=t_size, bbox=True)

                for j, (im, b) in enumerate(rescaled_images):  # hw
                    # opt out too small boxes
                    if b[2] < 8. or b[3] < 8.:
                        continue
                    row_new = row.copy()
                    r_sub, c_sub, _ = im.shape
                    # zero padding the cropped images
                    if r_sub < t_size or c_sub < t_size:
                        x_resized = zero_padding(im, t_size)
                    else:
                        x_resized = im.copy()
                    # update annotation for the cropped image
                    fname = f"{str(row['fn_new']).split('.')[0]}_{i}_{j}.jpeg"
                    row_new['fn_new'] = fname
                    row_new['label'] = str(label)
                    row_new['bbox'] = ' '.join([str(int(c)) for c in b])
                    df_new = df_new.append(row_new).reset_index(drop=True)
                    cv2.imwrite(fname, x_resized)
    df_new.drop(index=index_to_del, inplace=True)
    return df_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("orig_im_path", help="text file containing original image path")
    parser.add_argument("resized_im_path", help="resized image path")
    parser.add_argument("json_path", help="json format annotation path")
    parser.add_argument("df_path", help="annotation data frame path")
    parser.add_argument("target_size", help="target image size: 300/512")

    args = parser.parse_args()

    if not os.path.exists(args.resized_im_path):
        os.makedirs(args.resized_im_path)

    # target size
    t_size = int(args.target_size)

    # make annotation dictionary with original image and bboxes
    print('Making annotations from json file...')
    annotations = json.load(open(Path(args.json_path)/'train.json'))
    vals = json.load(open(Path(args.json_path)/'valid.json'))
    train_anno = make_anno_dict(annotations)
    train_bbox_multi_df = prepare_dataset(train_anno, args.resized_im_path)
    val_anno = make_anno_dict(vals)
    val_bbox_multi_df = prepare_dataset(val_anno, args.resized_im_path)
    print('Annotations done!')

    # resize images by cropping/zero padding
    print('Resizing images...')
    # '/data/xhan/Dropbox/USF MantaRay Data Share/Image Training Data'
    train_df = crop_and_pad(train_bbox_multi_df, t_size)
    val_df = crop_and_pad(val_bbox_multi_df, t_size)
    print('Image resizing done!')
    print('Annotations updated!')

    # write resized boxes to csv for future use
    print('Saving annotations to csv files...')
    train_df.to_csv(Path(args.df_path)/f'train_anno_{t_size}_crop.csv', index=False)
    val_df.to_csv(Path(args.df_path)/f'val_anno_{t_size}_crop.csv', index=False)
    print('Preprocessing completed. Exit...')
