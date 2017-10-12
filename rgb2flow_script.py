from glob import glob
from utils import ImageUtils
import os
import yaml
import h5py
import io
import numpy as np
import multiprocessing
from PIL import Image
import cv2

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

rootDir = config['ucf_rootDir']
flowDir = config['ucf_flowDir']
datasetDir = config['ucf_dataset']

f = h5py.File(datasetDir, 'w')
flow = f.create_group('flow')
rgb = f.create_group('rgb')

def Convert(actionPath, example, action):
    print("Started: ", example)
    examplePath = os.path.join(actionPath, example)

    frames = []
    images = []
    compressedFlowImages = []
    for frame in sorted(glob(examplePath + "/*.jpg")):
        im = open(frame, 'rb').read()
        frames.append(im)
        images.append(np.array(Image.open(io.BytesIO(im))))
        # img = Image.open(io.BytesIO(im))

    if not os.path.exists(os.path.join(flowDir, action, example)):
        os.makedirs(os.path.join(flowDir, action, example))

    flowFrames = ImageUtils.ComputeOpticalFlow(np.array(images), os.path.join(flowDir, action, example)) #.transpose(0, 3, 1, 2)

    for i, ff in enumerate(flowFrames):
        r, buf = cv2.imencode('.jpg', ff)
        compressedFlowImages.append(buf.tostring())

    rgb.create_dataset(example, data=frames)
    flow.create_dataset(example, data=compressedFlowImages)

    print(example, " Done :)")

for action in sorted(os.listdir(rootDir)):
    actionPath = os.path.join(rootDir, action)
    if os.path.isdir(actionPath):
        num_cores = multiprocessing.cpu_count() - 1
        # Parallel(n_jobs=num_cores)(delayed(Convert)(actionPath, example, action) for example in os.listdir(os.path.join(rootDir, action)))
        for example in sorted(os.listdir(os.path.join(rootDir, action))):
            Convert(actionPath, example, action)

