import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from ..transform.augmentation import normalize
from ..transform import __imgtools__, __boxtools__
from src.config.cuda_cfg import device


class multiBboxDataset(Dataset):
    def __init__(self, root, dataset, sz,
                 transform=None, target_transform=None):
        self.root = Path(root)
        self.dataset = self.make_anno_dict(dataset)
        self.sz = sz
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                        'sofa', 'train', 'tvmonitor', 'background']
        self.class_dict = {i: class_name for i, class_name in
                           enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image = self.get_image(i)
        labels = self.dataset[i]['labels']
        boxes = self.dataset[i]['boxes']  # hw

        if self.transform:  # augmentation
            image, boxes, labels = self.transform(image, boxes, labels)
        else:
            image = normalize(image)
            image = image.transpose(2, 0, 1)

        boxes = torch.from_numpy(boxes).float().to(device, non_blocking=True)
        labels = torch.from_numpy(labels).float().to(device, non_blocking=True)

        # convert boxes to center format and percentage coords
        boxes = __boxtools__.hw_center(boxes)
        boxes, _ = __boxtools__.to_percent_coords(image.shape[1:], boxes)

        locations, labels = self.target_transform(boxes, labels)

        return image, labels, locations  # center format

    def get_image(self, index):
        image_path = str(self.root / self.dataset[index]['img_name'])
        image = __imgtools__.load_image(image_path)
        return image

    def make_anno_dict(self, df):
        out = []
        for row in df.values:
            anno_dict = dict()
            anno_dict['img_name'] = row[1].split('/')[-1]
            anno_dict['labels'] = np.array([int(x) for x in row[2].split()])
            bbs = row[4].split()
            anno_dict['boxes'] = np.array(
                [bbs[i * 4:i * 4 + 4] for i in range(int(len(bbs) / 4))],
                dtype=np.float)
            out.append(anno_dict)
        return out
