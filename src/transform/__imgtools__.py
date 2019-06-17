import os
import cv2
import json
import math
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

from src.transform.__boxtools__ import (make_bb_px, to_bb, bb_hw_numpy,
                                        hw_bb_numpy)


# Read images
def load_image(img_path):
    try:
        img = cv2.imread(str(img_path)).astype(np.float32) / 255
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(e)
        print(str(img_path))


# Show image with matplotlib
def draw_outline(plt_object, lw):
    """Draw outline to make text and rectangle visible"""
    plt_object.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b):
    """Draw bounding box.

    Note: patches.Rectangle(left-bottom(x, y), width, height, ...)
    """
    patch = ax.add_patch(patches.Rectangle(
        b[:2], b[2], b[3], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, font_sz=14):
    """Show class label"""
    text = ax.text(*xy, txt,
                   verticalalignment='top', color='white', fontsize=font_sz,
                   weight='bold')
    draw_outline(text, 1)


def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    return ax


def show_by_index(idx: int, annotations: dict):
    img_path = str(IMG_PATH / f'{idx:06d}.jpg')
    img = load_image(img_path)

    if idx in annotations.keys():
        print(img_path)
        ax = show_img(img, figsize=(12, 12))
        for item in annotations[idx]:
            b = item[1]
            print(
                f'Bounding box:\nX: {b[0]}\nY: {b[1]}\nWidth: {b[2]}\nHeight: '
                f'{b[3]}')
            draw_rect(ax, b)
            draw_text(ax, b[:2], categories[item[0]])


def resize_img_and_write(img_list, resized_im_path, new_size):
    """Resize the original images and write the resized images
    to a new directory.

    :param IMG_PATH: original image path
    :param NEW_PATH: resized image path
    :param new_size: target image size
    :return: None
    """
    for path in tqdm(img_list):
        img = cv2.imread(str(path))
        x_resized = cv2.resize(img, (int(new_size), int(new_size)))
        fname = str(path).split('/')[-1]
        if not os.path.isdir(resized_im_path):
            os.makedirs(resized_im_path)
        cv2.imwrite(str(Path(resized_im_path)/fname), x_resized)
    tqdm.write(f'Successfully resize images. New path: {orig_im_paths}')


def resize_bbox(orig_shape, bbox, target_size):
    """Resize bounding boxes accordingly.

    :param orig_shape: shape of original image
    :param bbox: bounding boxes
    :param target_size: size of the resized image
    :return: new boxes (corner format)
    """
    Y = make_bb_px(bbox, orig_shape)
    y_resize = cv2.resize(Y, target_size)
    y = to_bb(y_resize)
    y[0], y[2] = (np.clip(y[0], 0, target_size[0]),
                  np.clip(y[2], 0, target_size[0]))
    y[1], y[3] = (np.clip(y[1], 0, target_size[1]),
                  np.clip(y[3], 0, target_size[1]))
    return y


def crop_imgs(img, y=None, size=300, bbox=False):
    h, w, _ = img.shape
    rescaled_images = [(img, h, w)]
    if bbox and y is not None:
        Y = make_bb_px(y, (h, w))

    # get crops
    hs = itertools.product([i * size for i in range(int(h / size) + 1)],
                           [min((i + 1) * size, h) for i in
                            range(int(h / size) + 1)])
    ws = itertools.product([i * size for i in range(int(w / size) + 1)],
                           [min((i + 1) * size, w) for i in
                            range(int(w / size) + 1)])
    hs_new = [h for h in hs if h[0] != h[1] and h[1] - size <= h[0] < h[1]]
    ws_new = [w for w in ws if w[0] != w[1] and w[1] - size <= w[0] < w[1]]
    all_sub_img = list(itertools.product(hs_new, ws_new))

    # make crops
    if bbox:
        rescaled_images = [(img[h[0]:h[1], w[0]:w[1]],
                            bb_hw_numpy(to_bb(Y[h[0]:h[1], w[0]:w[1]]))) for
                           h, w in all_sub_img]
    else:
        rescaled_images = [(img[h[0]:h[1], w[0]:w[1]], (h, w)) for h, w in
                           all_sub_img]

    return rescaled_images


def zero_padding(img, size):
    r, c, _ = img.shape
    padding = np.zeros((size, size, 3)).astype(np.float32)
    padding[:r, :c, :] = img.astype(np.float32)

    return padding
