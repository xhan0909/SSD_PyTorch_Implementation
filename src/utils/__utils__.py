import time
import torch
import itertools
import numpy as np
from typing import *
from collections import defaultdict

from src.config.cuda_cfg import device


def extract_metadata(json_path: str):
    """Make dictionaries:
    bounding boxes: {img_id:[(cat_id, bbox), (cat_id, bbox), ...]}
    categories: {id: category_name}
    """
    anno_json = json.load(open(json_path))
    anno_dict = defaultdict(list)
    categories = {d['id']: d['name'] for d in anno_json['categories']}
    for anno in anno_json['annotations']:
        if not anno['ignore']:
            bb = np.array(anno['bbox'])
            anno_dict[anno['image_id']].append(
                (anno['category_id'], bb))
    return anno_dict, categories


def even_mults(start: float, stop: float, n: int) -> np.ndarray:
    """Build log-stepped array from `start`  to `stop` in `n` steps evenly."""
    mult = stop / start
    step = mult ** (1 / (n - 1))

    return np.array([start * (step ** i) for i in range(n)])


def lr_range(lr: Union[float, slice], n) -> np.ndarray:
    """Build differential learning rates from `lr`."""
    if not isinstance(lr, slice):
        return lr
    if lr.start:
        lrs = even_mults(lr.start, lr.stop, n)
    else:
        lrs = [lr.stop / 10] * (n - 1) + [lr.stop]

    return np.array(lrs)


def get_triangular_lr(lr_optimal, div_factor, iterations):
    """
    Triangular learning rate.

    :param lr_optimal: optimal learning rate found by LR finder
    :param div_factor: division factor
    :param iterations: number of iterations
    :return: an array of learning rates
    """
    iter1 = int(0.45 * iterations)
    iter2 = int(1.0 * iter1)
    iter3 = iterations - iter1 - iter2

    lr_low = lr_optimal/div_factor
    lr_high = lr_optimal

    delta_1 = (lr_high - lr_low) / iter1
    delta_2 = (lr_high - lr_low) / (iter2 - 1)
    lrs1 = [lr_low + i * delta_1 for i in range(iter1)]
    lrs2 = [lr_high - i * delta_2 for i in range(0, iter2)]
    delta_3 = (lrs2[-1] - lr_low / 100) / iter3
    lrs3 = [lrs2[-1] - i * delta_3 for i in range(1, iter3 + 1)]

    return lrs1 + lrs2 + lrs3


def cosine_segment(start_lr, end_lr, iterations):
    i = np.arange(iterations)
    c_i = 1 + np.cos(i*np.pi/iterations)
    return end_lr + (start_lr - end_lr)/2 * c_i


def get_cosine_triangular_lr(lr_optimal, div_factor, iterations):
    """
    Triangular learning rate with cosine affine.

    :param lr_optimal: optimal learning rate found by LR finder
    :param div_factor: division factor
    :param iterations: number of iterations
    :return: an array of learning rates
    """
    min_start, min_end = lr_optimal / div_factor, lr_optimal / (div_factor*1e4)
    iter1 = int(0.3*iterations)
    iter2 = iterations - iter1
    segs = [cosine_segment(min_start, lr_optimal, iter1),
            cosine_segment(lr_optimal, min_end, iter2)]
    return np.concatenate(segs)


def get_triangular_mom(mom_low=0.85, mom_high=0.95, iterations=50):
    """
    Triangular momentum.

    :param mom_low: lower bound momentum
    :param mom_high: higher bound momentum
    :param iterations: number of iterations
    :return: an array of momentum values
    """
    iter1 = int(0.45 * iterations)
    iter2 = int(1.0 * iter1)
    iter3 = iterations - iter1 - iter2

    delta_1 = (mom_high - mom_low) / iter1
    delta_2 = (mom_high - mom_low) / (iter2 - 1)
    moms1 = [mom_high - i * delta_1 for i in range(iter1)]
    moms2 = [mom_low + i * delta_2 for i in range(0, iter2)]
    moms3 = [moms2[-1] for i in range(1, iter3 + 1)]

    return moms1 + moms2 + moms3


def save_model(m, p):
    torch.save(m.state_dict(), p)


def load_model(m, p):
    checkpoint = torch.load(p, map_location=device)
    m.load_state_dict(checkpoint)
    del checkpoint
    # torch.cuda.empty_cache()


def set_trainable_attr(m, b=True):
    for p in m.parameters():
        p.requires_grad = b


def unfreeze_top_model(model, l):
    top_model = model.top_model
    set_trainable_attr(top_model[l])


def unfreeze_vgg_extra(model, l):
    vgg_extra = model.extra2
    set_trainable_attr(vgg_extra[:l])


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]

        return interval
