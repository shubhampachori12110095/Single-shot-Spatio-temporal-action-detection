import torch
from ActionYolo import ActionYolo
import numpy as np
import pdb
import torch.nn.functional as F

class Inference():

    def __init__(self, model_path, num_classes=24, num_boxes=2):
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.net = ActionYolo(num_class=num_classes, bbox_num=num_boxes).cuda()
        self.net.load_state_dict(torch.load(model_path))


    def detect(self, rgb_frames, flow_frames, vis=False):
        output = self.net.forward(rgb_frames, flow_frames)

        class_confidence = output[:, :self.num_classes, :, :]
        region_confidence = output[:, self.num_classes : self.num_classes + self.num_boxes, :, :]
        predicted_bbox = output[:, self.num_classes + self.num_boxes:, :, :]
        for ff in range(len(rgb_frames)):

            class_probablties = F.softmax(class_confidence[ff])

            pdb.set_trace()

            action, cell_x, cell_y = np.argwhere(x == np.argmax(class_probablties.data.cpu().numpy()))

            box_index = np.argmax(region_confidence[class_probablties, :, cell_x, cell_y].data.cpu().numpy())

            bbox = predicted_bbox[class_probablties, box_index*4:box_index*4+4, cell_x, cell_y].data.cpu().numpy()

            print(action, cell_x, cell_y, box_index, bbox)

