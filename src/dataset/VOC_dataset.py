import cv2
import torch
import numpy as np
from ..transform import __imgtools__, __boxtools__


class multiBboxDataset(Dataset):
    def __init__(self, root, dataset, transform=False, target_transform=None,
                 sz=img_size, is_test=False):
        self.root = Path(root)
        self.dataset = self.make_anno_dict(dataset)
        self.sz = sz
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                        'sofa', 'train', 'tvmonitor', 'background']
        self.class_dict = {i: class_name for i, class_name in
                           enumerate(self.classes)}
        self.iscuda = torch.cuda.is_available()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image = self.get_image(i)
        labels = self.dataset[i]['labels']
        boxes = self.dataset[i]['boxes']
        boxes = torch.from_numpy(np.array(boxes)).float()
        labels = torch.from_numpy(np.array(labels)).float()
        if self.iscuda:
            boxes = boxes.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        boxes = __boxtools__.hw_bb(boxes)
        boxes, _ = __boxtools__.to_percent_coords(image.shape[:2], boxes)
        locations, labels = self.target_transform(boxes, labels)
        image = normalize(image)

        return image.transpose(2, 0, 1), labels, locations

    def get_image(self, index):
        image_path = str(self.root / self.dataset[index]['img_name'])
        image = __imgtools__.load_image(image_path)
        return image

    def make_anno_dict(self, df):
        out = []
        for row in df.values:
            anno_dict = dict()
            anno_dict['img_name'] = row[0].split('/')[-1]
            anno_dict['labels'] = np.array([int(x) for x in row[1].split()])
            bbs = row[3].split()
            anno_dict['boxes'] = np.array(
                [bbs[i * 4:i * 4 + 4] for i in range(int(len(bbs) / 4))],
                dtype=np.float)
            out.append(anno_dict)
        return out
