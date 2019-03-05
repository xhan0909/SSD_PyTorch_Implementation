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
