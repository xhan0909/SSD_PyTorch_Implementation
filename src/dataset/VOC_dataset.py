import cv2
import numpy as np

from ..transform import __imgtools__, __boxtools__, augmentation


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
                        'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                        'background']
        self.class_dict = {i: class_name for i, class_name in
                           enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image = self.get_image(i)
        labels = self.dataset[i]['labels']
        boxes = self.dataset[i]['boxes']
        if self.transform:
            image, boxes = self.tsfm(image, boxes, self.sz)
        if self.target_transform:
            if not isinstance(boxes, np.ndarray):
                boxes = np.array(boxes)
            locations, labels = self.target_transform(boxes, labels)
        image = __boxtools__.normalize(image)
        return image.transpose(2, 0, 1), labels, locations

    def tsfm(self, x, bbox, size):
        random_degree = (np.random.random() - .50) * 20
        Y = []
        for b in bbox:
            Y.append(make_bb_px(b, x))

        # rotate
        x = cv2.resize(x, (size, size))
        x_rot = augmentation.rotate_cv(x, random_degree)
        Y_rot = [None] * len(Y)
        for i, y in enumerate(Y):
            y = cv2.resize(y, (size, size))
            Y_rot[i] = augmentation.rotate_cv(y, random_degree, bbox=True)

        # random flip
        if np.random.random() > .5:
            x_flip = np.fliplr(x_rot).copy()
            Y_flip = []
            for i, y in enumerate(Y_rot):
                Y_flip.append(__boxtools__.bb_hw(
                    __boxtools__.to_bb(np.fliplr(y).copy())))
            _, Y_flip, _ = __boxtools__.to_percent_coords(x_flip,
                                                          np.array(Y_flip))
            return x_flip, Y_flip

        Y_rot = np.array([__boxtools__.bb_hw(
            __boxtools__.to_bb(y)) for y in Y_rot])
        _, Y_rot, _ = __boxtools__.to_percent_coords(x_rot, Y_rot)
        return x_rot, Y_rot

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
            bbs = row[2].split()
            anno_dict['boxes'] = np.array(
                [bbs[i * 4:i * 4 + 4] for i in range(int(len(bbs) / 4))],
                dtype=np.float)
            out.append(anno_dict)
        return out
