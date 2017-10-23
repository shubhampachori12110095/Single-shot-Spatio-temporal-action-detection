import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class swish(nn.Module):

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)
