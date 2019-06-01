import torch
import numpy as np
import torch.nn.functional as F

from ..utils import __utils__
from ..transform import __boxtools__
from ..transform.augmentation import normalize
from src.config.cuda_cfg import device


class Predictor:
    def __init__(self, net, priors, center_variance=0.1,
                 size_variance=0.2, iou_threshold=.5, candidate_size=200,
                 sigma=.5, nms_type='hard'):
        self.net = net
        self.priors = priors
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.candidate_size = candidate_size
        self.sigma = sigma
        self.nms_type = nms_type
        self.timer = __utils__.Timer()
        self.device = device
        self.net.to(self.device)
        self.net.eval()

    def predict(self, image, top_k=-1, prob_threshold=0.5, score_threshold=.3):
        """

        :param image: image to be predicted (normalized)
        :param top_k: max number of boxes to keep
        :param prob_threshold: probability threshold for hard nms
        :param score_threshold: score threshold for soft nms
        :return: center format boxes, labels, and confidence scores
        """
        # this version of hard_nms is slower on GPU, so we move data to CPU.
        cpu_device = torch.device("cpu")
        _, height, width = image.shape

        # image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            self.timer.start()
            confidences, locations = self.net(image.float())
            print("Inference time: ", self.timer.end())
        locations = torch.sigmoid(locations).squeeze().to(cpu_device)
        confidences = confidences.squeeze().to(cpu_device)
        confidences = F.softmax(confidences, dim=-1)
        # confidences = torch.sigmoid(confidences)

        # convert boxes
        boxes = __boxtools__.convert_locations_to_boxes(
            locations,  # center format
            self.priors,  # center format
            self.center_variance,
            self.size_variance)

        picked_box_probs = []
        picked_labels = []

        for class_index in range(1, confidences.size(1)):  # no background (0)
            probs = confidences[..., class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = __boxtools__.nms(box_probs,
                                         iou_threshold=self.iou_threshold,
                                         score_threshold=score_threshold,
                                         top_k=top_k,
                                         candidate_size=self.candidate_size,
                                         sigma=self.sigma,
                                         nms_type=self.nms_type)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        if not picked_box_probs:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        return (picked_box_probs[:, :4], torch.Tensor(picked_labels),
                picked_box_probs[:, -1])
