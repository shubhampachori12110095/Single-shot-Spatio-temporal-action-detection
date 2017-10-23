from scipy.io import loadmat
import os
from skimage import transform
import io

from torch.utils.data import Dataset, DataLoader
from glob import glob
import h5py
import numpy as np

import ActionYolo
import pdb
from PIL import Image
import cv2
import torch
from torch.autograd import Variable

class UCFDataSet(Dataset):

    def __init__(self, datasetFile, matFile, transform=None, subsample=False):

        self.datasetFile = datasetFile
        self.subsample = subsample
        self.annotations = loadmat(matFile)
        self.transform = transform
        self.dataset = None
        self.output_size = 448

    def __len__(self):
        return len(self.annotations['annot'][0])

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')

        example_name = self.annotations['annot'][0][idx][1][0]

        action = self.annotations['annot'][0][idx][2][0][0][2][0][0]
        startFrame = self.annotations['annot'][0][idx][2][0][0][1][0][0]
        endFrame = self.annotations['annot'][0][idx][2][0][0][0][0][0]

        # bboxes (x1, y2, w, h) in other words upper left corner, width and height
        bboxes = self.annotations['annot'][0][idx][2][0][0][3]
        framesCount = self.annotations['annot'][0][idx][0][0]

        encoded_rgb_frames = self.dataset['rgb'][example_name.split('/')[-1]]
        encoded_flow_frames = self.dataset['flow'][example_name.split('/')[-1]]

        rgb_frames = np.array([np.array(Image.open(io.BytesIO(im)), dtype=float) for im in encoded_rgb_frames], dtype=float).transpose(0, 3, 1, 2)
        flow_frames = np.array([np.array(cv2.imdecode(np.fromstring(ff, np.uint8), 1)[..., :2], dtype=float) for ff in encoded_flow_frames], dtype=float).transpose(0, 3, 1, 2)

        if self.subsample:
            idx = np.random.choice(range(startFrame-1, endFrame), 8)
            rgb_frames = rgb_frames[idx]
            flow_frames = flow_frames[idx]
            bboxes = bboxes[idx - startFrame + 1]

        rgb_frames = (rgb_frames - np.mean(rgb_frames, axis=(2, 3)).reshape(*rgb_frames.shape[:2], 1, 1))\
                     / (np.std(rgb_frames, axis=(2, 3)).reshape(*rgb_frames.shape[:2], 1, 1) + 1e-7)

        flow_frames = (flow_frames - np.mean(flow_frames, axis=(2, 3)).reshape(*flow_frames.shape[:2], 1, 1))\
                      / (np.std(flow_frames, axis=(2, 3)).reshape(*flow_frames.shape[:2], 1, 1) + 1e-7)

        sample = {
                  'frames': rgb_frames,
                  'flowFrames': flow_frames,
                  'action': np.int32(action),
                  'startFrame': np.int32(startFrame),
                  'endFrame': np.int32(endFrame),
                  'bbox': np.array(bboxes, dtype=float)
                  }

        if self.transform:
            sample = self.transform(sample)

        print(example_name.split('/')[-1])

        return sample

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        old_w = 1.0 * 240
        old_h = 1.0 * 320

        scale_x = self.output_size / old_w
        scale_y = self.output_size / old_h

        bbox = sample['bbox']

        bbox[:, 0] = bbox[:, 0] * scale_x
        bbox[:, 2] = bbox[:, 2] * scale_x
        bbox[:, 1] = bbox[:, 1] * scale_y
        bbox[:, 3] = bbox[:, 3] * scale_y

        sample['bbox'] = bbox

        return sample
