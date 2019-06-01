from ..transform.augmentation import *


class TrainTransform:
    def __init__(self, size, mean=0, std=1.0):
        """
        :param size: the size the of final image
        :param mean: mean pixel value per channel
        :param std: standard deviation
        """
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            # ConvertFromInts(),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomFlip(),
            # RandomRotate(),
            # RandomSampleCrop(),
            # Resize(self.size),
            # SubtractMeans(self.mean),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class ValidTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        """
        :param size: the size the of final image
        :param mean: mean pixel value per channel
        :param std: standard deviation
        """
        self.mean = mean
        self.std = std
        self.size = size
        self.transform = Compose([
            # SubtractMeans(mean),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)
