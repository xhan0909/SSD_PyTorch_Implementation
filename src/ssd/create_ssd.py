import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d

from .ssd_model import SSDNet
from .ssd_predict import Predictor
from src.config.cuda_cfg import device


def create_vgg_ssd(num_classes):
    net = SSDNet(num_classes=num_classes).to(device)

    return net


def create_vgg_ssd_predictor(net, priors, candidate_size=200,
                             nms_method='hard', iou_threshold=0.5,
                             sigma=0.5):
    predictor = Predictor(net, priors,
                          center_variance=0.1,
                          size_variance=0.2,
                          iou_threshold=iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          nms_type=nms_method)
    return predictor
