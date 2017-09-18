import numpy as np
import torch
import torchvision
import  torchvision.transforms  as transforms
import torch.nn as nn
import torch.nn.functional as F


class ActionYolo(nn.Module):
    def __init__(self, config):
        super(ActionYolo, self).__init__()
        b = config['B']
        c = config['C']

        # Spatial CNN
        self.spatilConv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.spatilConv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.spatilConv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.spatilConv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.spatilConv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.spatilConv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.spatilConv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.spatilConv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.spatilConv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.spatilConv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.spatilConv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.spatilConv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.spatilConv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.spatilConv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.spatilConv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.spatilConv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.spatilConv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.spatilConv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.spatilConv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.spatilConv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.spatilConv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.spatilConv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.spatilConv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.spatilConv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)

        # Temporal CNN
        self.temporalConv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.temporalConv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.temporalConv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.temporalConv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.temporalConv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.temporalConv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.temporalConv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.temporalConv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.temporalConv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.temporalConv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.temporalConv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.temporalConv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.temporalConv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.temporalConv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.temporalConv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.temporalConv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.temporalConv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.temporalConv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.temporalConv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.temporalConv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.temporalConv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.temporalConv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.temporalConv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.temporalConv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)

        # Aggregation Layer
        self.conv25 = nn.Conv2d(in_channels=2048, out_channels=b*5 + c, kernel_size=1, stride=1, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, rgb, opticalFlow):

        # Forward Spatial
        x = self.pool(F.leaky_relu(self.spatilConv1(rgb), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.spatilConv2(x), negative_slope=0.1))
        x = F.leaky_relu(self.spatilConv3(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv4(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv5(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv6(x), negative_slope=0.1)
        x = self.pool(F.leaky_relu(self.spatilConv7(x), negative_slope=0.1))
        x = F.leaky_relu(self.spatilConv8(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv9(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv10(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv11(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv12(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv13(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv14(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv15(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv16(x), negative_slope=0.1)
        x = self.pool(F.leaky_relu(self.spatilConv17(x), negative_slope=0.1))
        x = F.leaky_relu(self.spatilConv18(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv19(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv20(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv21(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv22(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv23(x), negative_slope=0.1)
        x = F.leaky_relu(self.spatilConv24(x), negative_slope=0.1)

        # Forward Temporal
        y = self.pool(F.leaky_relu(self.temporalConv1(opticalFlow), negative_slope=0.1))
        y = self.pool(F.leaky_relu(self.temporalConv2(y), negative_slope=0.1))
        y = F.leaky_relu(self.temporalConv3(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv4(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv5(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv6(y), negative_slope=0.1)
        y = self.pool(F.leaky_relu(self.temporalConv7(y), negative_slope=0.1))
        y = F.leaky_relu(self.temporalConv8(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv9(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv10(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv11(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv12(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv13(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv14(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv15(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv16(y), negative_slope=0.1)
        y = self.pool(F.leaky_relu(self.temporalConv17(y), negative_slope=0.1))
        y = F.leaky_relu(self.temporalConv18(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv19(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv20(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv21(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv22(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv23(y), negative_slope=0.1)
        y = F.leaky_relu(self.temporalConv24(y), negative_slope=0.1)

        output = F.linear(self.conv25(torch.cat((x, y))))

        return output















