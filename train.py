import torch
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from ActionYolo import ActionYolo
from detectionLoss import DetectionLoss
from torch.autograd import Variable
from torchvision import transforms

class Trainer():
    def __init__(self, opts):
        self.opts = opts

    def train(self):

        X_spatial, X_temporal, target = dataloader()

        actionYolo = ActionYolo()

        # hyper-params
        learning_rate = 0.001
        momentum = 0.9

        criterion = DetectionLoss()
        optimizer = torch.optim.SGD(actionYolo.parameters(), lr=learning_rate, momentum=momentum)

        output = actionYolo.forward(X_spatial, X_temporal)
        loss = criterion(output, target)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()