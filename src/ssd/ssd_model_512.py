import torch
import torchvision
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .l2norm import L2Norm


class SSDNet(nn.Module):
    def __init__(self, num_classes, im_shape, is_test=False):
        super(SSDNet, self).__init__()
        self.num_classes = num_classes
        self.im_shape = im_shape
        self.is_test = is_test

        # 1: vgg16 (0-22), 512x64x64
        vgg16 = models.vgg16(pretrained=True)
        # apply vgg up to conv4_3 relu
        for param in vgg16.parameters():
            param.requires_grad = False
        layers = vgg16.features[:23]
        self.top_model = nn.Sequential(*layers)
        # for param in self.top_model.parameters():
        #     param.requires_grad = False

        self.l2norm = L2Norm(512, 20)
        # self.l2norm = nn.BatchNorm2d(512)
        self.cls1_conv = nn.Conv2d(512, 4 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg1_conv = nn.Conv2d(512, 4 * 4, kernel_size=3, stride=1,
                                   padding=1)

        # 2: vgg16 (23-29), 1024x32x32
        layers_2 = vgg16.features[23:30]
        self.extra2 = nn.Sequential(
            *layers_2,  # apply vgg up to fc7
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU()
        )
        #         self.bn2 = nn.BatchNorm2d(1024)
        self.cls2_conv = nn.Conv2d(1024, 6 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg2_conv = nn.Conv2d(1024, 6 * 4, kernel_size=3, stride=1,
                                   padding=1)

        # 3: 512x16x16
        self.extra3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        #         self.bn3 = nn.BatchNorm2d(512)
        self.cls3_conv = nn.Conv2d(512, 6 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg3_conv = nn.Conv2d(512, 6 * 4, kernel_size=3, stride=1,
                                   padding=1)
        self.init_weights(self.extra3, self.cls3_conv, self.reg3_conv)

        # 4: 256x8x8
        self.extra4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        #         self.bn4 = nn.BatchNorm2d(256)
        self.cls4_conv = nn.Conv2d(256, 6 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg4_conv = nn.Conv2d(256, 6 * 4, kernel_size=3, stride=1,
                                   padding=1)
        self.init_weights(self.extra4, self.cls4_conv, self.reg4_conv)

        # 5: 256x4x4
        self.extra5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        #         self.bn5 = nn.BatchNorm2d(256)
        self.cls5_conv = nn.Conv2d(256, 6 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg5_conv = nn.Conv2d(256, 6 * 4, kernel_size=3, stride=1,
                                   padding=1)
        self.init_weights(self.extra5, self.cls5_conv, self.reg5_conv)

        # 6: 256x2x2
        self.extra6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        #         self.bn6 = nn.BatchNorm2d(256)
        self.cls6_conv = nn.Conv2d(256, 4 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg6_conv = nn.Conv2d(256, 4 * 4, kernel_size=3, stride=1,
                                   padding=1)
        self.init_weights(self.extra6, self.cls6_conv, self.reg6_conv)

        # 7: 256x1x1
        self.extra7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU()
        )
        #         self.bn6 = nn.BatchNorm2d(256)
        self.cls7_conv = nn.Conv2d(256, 4 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg7_conv = nn.Conv2d(256, 4 * 4, kernel_size=3, stride=1,
                                   padding=1)
        self.init_weights(self.extra7, self.cls7_conv, self.reg7_conv)

    def forward(self, X):
        confidences = []
        locations = []
        X = self.top_model(X)

        # 1
        X1 = self.l2norm(X)
        X1_cls = self.cls1_conv(X1)
        X1_reg = self.reg1_conv(X1)
        X1_conf, X1_location = self.compute_header(X1_cls, X1_reg)
        confidences.append(X1_conf)
        locations.append(X1_location)
        # 2
        X2 = self.extra2(X1)
        X2_cls = self.cls2_conv(X2)
        X2_reg = self.reg2_conv(X2)
        X2_conf, X2_location = self.compute_header(X2_cls, X2_reg)
        confidences.append(X2_conf)
        locations.append(X2_location)
        # 3
        X3 = self.extra3(X2)
        X3_cls = self.cls3_conv(X3)
        X3_reg = self.reg3_conv(X3)
        X3_conf, X3_location = self.compute_header(X3_cls, X3_reg)
        confidences.append(X3_conf)
        locations.append(X3_location)
        # 4
        X4 = self.extra4(X3)
        X4_cls = self.cls4_conv(X4)
        X4_reg = self.reg4_conv(X4)
        X4_conf, X4_location = self.compute_header(X4_cls, X4_reg)
        confidences.append(X4_conf)
        locations.append(X4_location)
        # 5
        X5 = self.extra5(X4)
        X5_cls = self.cls5_conv(X5)
        X5_reg = self.reg5_conv(X5)
        X5_conf, X5_location = self.compute_header(X5_cls, X5_reg)
        confidences.append(X5_conf)
        locations.append(X5_location)
        # 6
        X6 = self.extra6(X5)
        X6_cls = self.cls6_conv(X6)
        X6_reg = self.reg6_conv(X6)
        X6_conf, X6_location = self.compute_header(X6_cls, X6_reg)
        confidences.append(X6_conf)
        locations.append(X6_location)
        # 7
        X7 = self.extra6(X6)
        X7_cls = self.cls6_conv(X7)
        X7_reg = self.reg6_conv(X7)
        X7_conf, X7_location = self.compute_header(X7_cls, X7_reg)
        confidences.append(X7_conf)
        locations.append(X7_location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        return confidences, locations

    def compute_header(self, X_cls, X_reg):
        X_cls = X_cls.permute(0, 2, 3, 1).contiguous()
        X_conf = X_cls.view(X_cls.size(0), -1, self.num_classes)

        X_reg = X_reg.permute(0, 2, 3, 1).contiguous()
        X_location = X_reg.view(X_reg.size(0), -1, 4)

        return X_conf, X_location

    def init_weights(self, extra, cls_conv, reg_conv):
        extra.apply(self._xavier_init_)
        cls_conv.apply(self._xavier_init_)
        reg_conv.apply(self._xavier_init_)

    def _xavier_init_(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
