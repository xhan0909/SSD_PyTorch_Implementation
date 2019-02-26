import os
import cv2
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects


# Read images
def load_image(img_path):
    img = cv2.imread(str(img_path)).astype(np.float32) / 255
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


# Show image with matplotlib
def draw_outline(plt_object, lw):
    """Draw outline to make text and rectangle visible"""
    plt_object.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b):
    """Draw bounding box"""
    patch = ax.add_patch(patches.Rectangle(
        b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
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
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

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


# Image Augmentations
def normalize(im):
    """Normalize images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]


def denormalize(im):
    """Denormalize images."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im * imagenet_stats[1]) + imagenet_stats[0]


# For images
def crop(im, r, c, target_r, target_c):
    return im[r:r+target_r, c:c+target_c]


def random_crop(x, target_r, target_c):
    """ Returns a random crop"""
    r, c, *_ = x.shape
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(rand_r*(r - target_r)).astype(int)
    start_c = np.floor(rand_c*(c - target_c)).astype(int)
    return crop(x, start_r, start_c, target_r, target_c)


def rotate_cv(im, deg, bbox=False,
              mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r, c, *_ = im.shape
    M = cv2.getRotationMatrix2D((c/2, r/2), deg, 1)
    if bbox:
        return cv2.warpAffine(im, M, (c, r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im, M, (c, r), borderMode=mode,
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)


# For bounding boxes
def make_bb_px(y, x):
    """Makes an image of size x retangular bounding box"""
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


def tsfm_img_bb(x, bbox, size=224):
    """Make consistent transformation of an image and its bounding boxes."""
    random_degree = (np.random.random()-.50)*20
    Y = make_bb_px(bbox, x)

    # resize
    x = cv2.resize(x, (size, size))
    y = cv2.resize(Y, (size, size))
    # then rotate
    x = rotate_cv(x, random_degree)
    y = rotate_cv(y, random_degree, bbox=True)
    # then random flip
    if np.random.random() > 0.5:
        x = np.fliplr(x).copy()
        y = np.fliplr(y).copy()

    return x, bb_hw(to_bb(y))
