import torch
import torchvision
import torch.nn as nn
from torchvision import models


class SSDNet(nn.Module):
    def __init__(self, num_classes=21, im_shape=(300, 300), is_test=False):
        super(SSDNet, self).__init__()
        self.num_classes = num_classes
        self.im_shape = im_shape
        self.is_test = is_test
        # 1: vgg16 (0-22), 512x38x38
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad = False
        layers = vgg16.features[:23]  # all the way to Conv4-3
        self.top_model = nn.Sequential(*layers)
        self.bn1 = nn.BatchNorm2d(512)
        self.cls1_conv = nn.Conv2d(512, 4 * self.num_classes, kernel_size=2,
                                   stride=1, padding=1)
        self.reg1_conv = nn.Conv2d(512, 4 * 4, kernel_size=2, stride=1,
                                   padding=1)

        # 2: vgg16 (23-29), 1024x19x19
        layers_2 = vgg16.features[23:30]
        self.extra2 = nn.Sequential(
            *layers_2,  # Conv5-1, 5-2, 5-3
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        #         self.bn2 = nn.BatchNorm2d(1024)
        self.cls2_conv = nn.Conv2d(1024, 6 * self.num_classes, kernel_size=2,
                                   stride=1, padding=1)
        self.reg2_conv = nn.Conv2d(1024, 6 * 4, kernel_size=2, stride=1,
                                   padding=1)

        # 3: 512x10x10
        self.extra3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #         self.bn3 = nn.BatchNorm2d(512)
        self.cls3_conv = nn.Conv2d(512, 6 * self.num_classes, kernel_size=2,
                                   stride=1, padding=1)
        self.reg3_conv = nn.Conv2d(512, 6 * 4, kernel_size=2, stride=1,
                                   padding=1)

        # 4: 256x5x5
        self.extra4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        #         self.bn4 = nn.BatchNorm2d(256)
        self.cls4_conv = nn.Conv2d(256, 6 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg4_conv = nn.Conv2d(256, 6 * 4, kernel_size=3, stride=1,
                                   padding=1)

        # 5: 256x3x3
        self.extra5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        #         self.bn5 = nn.BatchNorm2d(256)
        self.cls5_conv = nn.Conv2d(256, 4 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg5_conv = nn.Conv2d(256, 4 * 4, kernel_size=3, stride=1,
                                   padding=1)

        # 6: 256x1x1
        self.extra6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        #         self.bn6 = nn.BatchNorm2d(256)
        self.cls6_conv = nn.Conv2d(256, 4 * self.num_classes, kernel_size=3,
                                   stride=1, padding=1)
        self.reg6_conv = nn.Conv2d(256, 4 * 4, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, X):
        confidences = []
        locations = []
        X = self.top_model(X)
        # 1
        X1 = self.bn1(X)
        # X1 = nn.Dropout2d(0.1)(X1_o)
        X1_cls = self.cls1_conv(X1)
        X1_reg = self.reg1_conv(X1)
        X1_conf = X1_cls.permute(0, 2, 3, 1).contiguous()
        X1_conf = X1_conf.view(X1_conf.size(0), -1, self.num_classes)
        X1_location = X1_reg.permute(0, 2, 3, 1).contiguous()
        X1_location = X1_location.view(X1_location.size(0), -1, 4)
        confidences.append(X1_conf)
        locations.append(X1_location)
        # 2
        X2 = self.extra2(X1)
        # X2 = nn.Dropout2d(0.1)(X2_o)
        X2_cls = self.cls2_conv(X2)
        X2_reg = self.reg2_conv(X2)
        X2_conf = X2_cls.permute(0, 2, 3, 1).contiguous()
        X2_conf = X2_conf.view(X2_conf.size(0), -1, self.num_classes)
        X2_location = X2_reg.permute(0, 2, 3, 1).contiguous()
        X2_location = X2_location.view(X2_location.size(0), -1, 4)
        confidences.append(X2_conf)
        locations.append(X2_location)
        # 3
        X3 = self.extra3(X2)
        # X3 = nn.Dropout2d(0.1)(X3_o)
        X3_cls = self.cls3_conv(X3)
        X3_reg = self.reg3_conv(X3)
        X3_conf = X3_cls.permute(0, 2, 3, 1).contiguous()
        X3_conf = X3_conf.view(X1_conf.size(0), -1, self.num_classes)
        X3_location = X3_reg.permute(0, 2, 3, 1).contiguous()
        X3_location = X3_location.view(X3_location.size(0), -1, 4)
        confidences.append(X3_conf)
        locations.append(X3_location)
        # 4
        X4 = self.extra4(X3)
        # X4 = nn.Dropout2d(0.1)(X4_o)
        X4_cls = self.cls4_conv(X4)
        X4_reg = self.reg4_conv(X4)
        X4_conf = X4_cls.permute(0, 2, 3, 1).contiguous()
        X4_conf = X4_conf.view(X4_conf.size(0), -1, self.num_classes)
        X4_location = X4_reg.permute(0, 2, 3, 1).contiguous()
        X4_location = X4_location.view(X4_location.size(0), -1, 4)
        confidences.append(X4_conf)
        locations.append(X4_location)
        # 5
        X5 = self.extra5(X4)
        # X5 = nn.Dropout2d(0.1)(X5_o)
        X5_cls = self.cls5_conv(X5)
        X5_reg = self.reg5_conv(X5)
        X5_conf = X5_cls.permute(0, 2, 3, 1).contiguous()
        X5_conf = X5_conf.view(X5_conf.size(0), -1, self.num_classes)
        X5_location = X5_reg.permute(0, 2, 3, 1).contiguous()
        X5_location = X5_location.view(X5_location.size(0), -1, 4)
        confidences.append(X5_conf)
        locations.append(X5_location)
        # 6
        X6 = self.extra6(X5)
        # X6 = nn.Dropout2d(0.1)(X6_o)
        X6_cls = self.cls6_conv(X6)
        X6_reg = self.reg6_conv(X6)
        X6_conf = X6_cls.permute(0, 2, 3, 1).contiguous()
        X6_conf = X6_conf.view(X6_conf.size(0), -1, self.num_classes)
        X6_location = X6_reg.permute(0, 2, 3, 1).contiguous()
        X6_location = X6_location.view(X6_location.size(0), -1, 4)
        confidences.append(X6_conf)
        locations.append(X6_location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        return confidences, locations
