import torch
import torch.nn as nn
from utils import Conv2d_BatchNorm
import torch.nn.parallel


class ActionYolo(nn.Module):
    def __init__(self, num_class, bbox_num=2, fusion='AVERAGE'):
        super(ActionYolo, self).__init__()
        self.b = bbox_num
        self.c = num_class
        self.image_size = 448

        self.spatialNet = nn.Sequential(
            nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear'),
            Conv2d_BatchNorm(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=64, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=192, out_channels=128, kernel_size=1),
            Conv2d_BatchNorm(in_channels=128, out_channels=256, kernel_size=3),
            Conv2d_BatchNorm(in_channels=256, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=512, kernel_size=1),
            Conv2d_BatchNorm(in_channels=512, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=512, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=512, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=512, kernel_size=1),
            Conv2d_BatchNorm(in_channels=512, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3, stride=2),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3)
        ).cuda()

        self.temporalNet = nn.Sequential(
            nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear'),
            Conv2d_BatchNorm(in_channels=2, out_channels=64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=64, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=192, out_channels=128, kernel_size=1),
            Conv2d_BatchNorm(in_channels=128, out_channels=256, kernel_size=3),
            Conv2d_BatchNorm(in_channels=256, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=256, kernel_size=1),
            Conv2d_BatchNorm(in_channels=256, out_channels=512, kernel_size=3),
            Conv2d_BatchNorm(in_channels=512, out_channels=512, kernel_size=1),
            Conv2d_BatchNorm(in_channels=512, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=512, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d_BatchNorm(in_channels=512, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=512, kernel_size=1),
            Conv2d_BatchNorm(in_channels=512, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3, stride=2),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3),
            Conv2d_BatchNorm(in_channels=1024, out_channels=1024, kernel_size=3)
        ).cuda()

        # Aggregation Layer
        self.conv25 = nn.Conv2d(in_channels=1024, out_channels=self.b*5 + self.c, kernel_size=1)

    def forward(self, rgb, opticalFlow):

        x = self.spatialNet.forward(rgb)
        y = self.temporalNet.forward(opticalFlow)

        output = self.conv25(0.5 * x + 0.5 * y)

        return output















