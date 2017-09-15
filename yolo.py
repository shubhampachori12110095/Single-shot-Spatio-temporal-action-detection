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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1)
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv25 = nn.Conv2d(in_channels=1024, out_channels=b*5 + c, kernel_size=1, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x), negative_slope=0.1))
        x = self.pool(F.leaky_relu(self.conv2(x), negative_slope=0.1))
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv6(x), negative_slope=0.1)
        x = self.pool(F.leaky_relu(self.conv7(x), negative_slope=0.1))
        x = F.leaky_relu(self.conv8(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv9(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv10(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv11(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv12(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv13(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv14(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv15(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv16(x), negative_slope=0.1)
        x = self.pool(F.leaky_relu(self.conv17(x), negative_slope=0.1))
        x = F.leaky_relu(self.conv18(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv19(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv20(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv21(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv22(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv23(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv24(x), negative_slope=0.1)
        x = F.linear(self.conv25(x))

    def loss(self, output, target):
        pass















