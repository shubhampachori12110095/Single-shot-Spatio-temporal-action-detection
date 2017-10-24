# Imports
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
from scipy.io import loadmat
from joblib import Parallel, delayed
import pdb
import argparse

# Global Vars
PARALLEL = False
parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true')
debug = parser.parse_args().debug

# Load Config File
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

# Load Annotations and Do mapping
annotations = loadmat(config['ucf_annotations'])
annot_mapping = {}
for idx in range(len(annotations['annot'][0])):
    example_name = annotations['annot'][0][idx][1][0]
    example_name = example_name.split('/')[-1]
    annot_mapping[example_name] = idx
print("Finished mapping")

# Initialize H5py tree structure
rootDir = config['ucf_rootDir']
flowDir = config['ucf_flowDir']
datasetDir = config['ucf_dataset']
tinyDatasetDir = config['ucf_tinyDataset']

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
test = f.create_group('test')
tiny_datset = h5py.File(tinyDatasetDir, 'w')

#Define Conversion Method
def Convert(actionPath, example, action, test_set, id, tiny_set):
    print("Started: ", example)
    examplePath = os.path.join(actionPath, example)

    frames = []
    images = []
    compressedFlowImages = []
    for frame in sorted(glob(examplePath + "/*.jpg")):
        im = open(frame, 'rb').read()
        frames.append(im)
        images.append(np.array(Image.open(io.BytesIO(im))))

    if not os.path.exists(os.path.join(flowDir, action, example)):
        os.makedirs(os.path.join(flowDir, action, example))

    flowFrames = ImageUtils.ComputeOpticalFlow(np.array(images), os.path.join(flowDir, action, example))
    for i, ff in enumerate(flowFrames):
        r, buf = cv2.imencode('.jpg', ff)
        compressedFlowImages.append(buf.tostring())

    if id in test_set:
        ex = test.create_group(example)
    else:
        ex = train.create_group(example)

    if debug:
        pdb.set_trace()

    ex.create_dataset("rgb", data=frames)
    ex.create_dataset("flow", data=compressedFlowImages)

    annots = ex.create_group("annot")

    if example not in annot_mapping:
        print("EXCEPTION: ", example)
        return

    example_id = annot_mapping[example]
    annots.create_dataset('action', data=annotations['annot'][0][example_id][2][0][0][2][0][0])
    annots.create_dataset('startFrame', data=annotations['annot'][0][example_id][2][0][0][1][0][0])
    annots.create_dataset('endFrame', data=annotations['annot'][0][example_id][2][0][0][0][0][0])
    annots.create_dataset('bboxes', data=annotations['annot'][0][example_id][2][0][0][3])

    if tiny_set:
        tiny_ex = tiny_datset.create_group(example)
        tiny_ex.create_dataset("rgb", data=frames)
        tiny_ex.create_dataset("flow", data=compressedFlowImages)
        tiny_annots = tiny_ex.create_group("annot")
        tiny_annots.create_dataset('action', data=annotations['annot'][0][example_id][2][0][0][2][0][0])
        tiny_annots.create_dataset('startFrame', data=annotations['annot'][0][example_id][2][0][0][1][0][0])
        tiny_annots.create_dataset('endFrame', data=annotations['annot'][0][example_id][2][0][0][0][0][0])
        tiny_annots.create_dataset('bboxes', data=annotations['annot'][0][example_id][2][0][0][3])

    print(example, ", IsTrain: ", id in test_set)

tiny_set = np.random.choice(range(24), 3, replace=True)
print(tiny_set)
for label, action in enumerate(sorted(os.listdir(rootDir))):
    actionPath = os.path.join(rootDir, action)
    if os.path.isdir(actionPath):
        if PARALLEL:
            num_cores = multiprocessing.cpu_count() - 1
            Parallel(n_jobs=num_cores)(delayed(Convert)(actionPath, example, action) for example in os.listdir(os.path.join(rootDir, action)))
        else:
            examples_files = sorted(os.listdir(os.path.join(rootDir, action)))
            # Do 2:1 Split
            test_set = np.random.choice(range(len(examples_files)), int(len(examples_files) / 3.0), replace=True)
            for id, example in enumerate(examples_files):
                Convert(actionPath, example, action, test_set, id, label in tiny_set)

