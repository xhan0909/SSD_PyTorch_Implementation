import cv2
import random
import numpy as np


def normalize(self, img):
    """Normalize images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]


def denormalize(im):
    """Denormalize images."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im * imagenet_stats[1]) + imagenet_stats[0]


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


def tsfm_img_bb(x, bbox, size=300):
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
