import os
import cv2
import torch
import argparse
import numpy as np
from typing import *

from src.utils.__utils__ import load_model
from src.transform.augmentation import normalize
from src.ssd.create_ssd import create_vgg_ssd_predictor
from src.transform.__boxtools__ import (generate_ssd_priors,
                                        center_hw, make_bb_px)
from src.transform.__imgtools__ import resize_bbox
from src.config.cuda_cfg import device


def get_test_img(image_path: str) -> List:
    """
    Get all test images and return a list of file paths.

    :param image_path: string, path for test images' directory
    :return: List, full path of each test image
    """
    images = os.listdir(image_path)
    images = list(map(lambda x: image_path + '/' + x, images))

    return images


def cut_4(img, r, c):
    crops = [img[:int(r / 2), :int(c / 2), :],
             img[int(r / 2):, :int(c / 2), :],
             img[:int(r / 2), int(c / 2):, :],
             img[int(r / 2):, int(c / 2):, :]]

    return crops


def make_prediction(im_paths: List, size: int,
                    priors: torch.Tensor, out_path: str):
    for path in im_paths:
        new_boxes = []
        orig_image = cv2.imread(path)
        resized = cv2.resize(orig_image, (size, size))
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        image = normalize(image)
        image = torch.from_numpy(image).permute([2, 0, 1]).float()

        # create predictor and make prediction
        predictor = create_vgg_ssd_predictor(model, priors,
                                             candidate_size=500,
                                             iou_threshold=config.iou_threshold,
                                             nms_method='hard')
        boxes, labels, probs = predictor.predict(image, top_k=2,
                                                 prob_threshold=config.prob_threshold)

        if len(boxes) > 0:
            for i in range(boxes.size(0)):
                box = center_hw(boxes[i, :])
                box = resize_bbox((size, size), box,
                                  (orig_image.shape[1], orig_image.shape[0]))
                if box.sum() == 0:
                    continue
                new_boxes.append((box, probs[i], labels[i]))

        # if nothing found under the threshold, do 4 crops and try again
        else:
            r, c, _ = orig_image.shape
            small_images = cut_4(orig_image, r, c)

            for i, img in enumerate(small_images):
                resized = cv2.resize(img, (size, size))
                image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                image = normalize(image)
                image = torch.from_numpy(image).permute([2, 0, 1]).float()

                # create predictor and make prediction
                predictor = create_vgg_ssd_predictor(model, priors.cpu(),
                                                     candidate_size=500,
                                                     iou_threshold=config.iou_threshold,
                                                     nms_method='hard')
                boxes, labels, probs = predictor.predict(image, top_k=1,
                                                         prob_threshold=config.prob_threshold+0.1)

                for j in range(boxes.size(0)):
                    box = center_hw(boxes[j, :])
                    box = resize_bbox((size, size), box,
                                      (img.shape[1], img.shape[0]))
                    if box.sum() == 0:
                        continue

                    if i == 0:
                        new_boxes.append((box, probs[j], labels[j]))
                    elif i == 1:
                        new_boxes.append((box + np.array(
                            [int(r / 2), 0, int(r / 2), 0]), probs[j], labels[j]))
                    elif i == 2:
                        new_boxes.append((box + np.array(
                            [0, int(c / 2), 0, int(c / 2)]), probs[j], labels[j]))
                    else:
                        new_boxes.append((box + np.array(
                            [int(r / 2), int(c / 2), int(r / 2), int(c / 2)]),
                                               probs[j], labels[j]))

        # save the image with boxes
        save_path = out_path + '/' + path.split('/')[-1]
        save_prediction(orig_image, new_boxes, save_path)


def save_prediction(img, boxes, save_path):
    categories = {0: 'background', 1: 'microfiber'}

    # text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .6
    font_color = (255, 0, 255)
    line_type = 2

    # draw prediction result
    for box, prob, label in boxes:
        cv2.rectangle(img, (int(box[1] - 10), int(box[0] - 10)),
                      (int(box[3] + 10), int(box[2] + 10)), (255, 255, 0), 1)
        label = f"{categories[label.item()]}: {prob.item():.4f}"
        cv2.putText(img,
                    label,
                    (int(box[1] - 5), int(box[2] + 35)),
                    font,
                    font_scale,
                    font_color,
                    line_type)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model path, *.pth")
    parser.add_argument("img_path", help="test image path (folder)")
    parser.add_argument("output_path", help="output image path (folder)")
    parser.add_argument("model_type", help="model type: ssd300/ssd512")
    args = parser.parse_args()

    # choose model type
    if args.model_type == 'ssd300':
        import src.config.prior_box_cfg_SSD300 as config
        from src.ssd.ssd_model import SSDNet
    elif args.model_type == 'ssd512':
        import src.config.prior_box_cfg_SSD512 as config
        from src.ssd.ssd_model_512 import SSDNet

    # Load model
    model = SSDNet(num_classes=config.num_classes,
                   im_shape=(config.image_size, config.image_size)).to(device)
    load_model(model, args.model_path)

    # Make prior boxes
    priors = generate_ssd_priors(config.specs,
                                 image_size=config.image_size, clip=True)
    priors = torch.from_numpy(priors).float()

    # make prediction
    im_paths = get_test_img(args.img_path)
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    make_prediction(im_paths, config.image_size, priors, args.output_path)
