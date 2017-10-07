import torch
import yaml
import pdb
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from ActionYolo import ActionYolo
from detectionLoss import DetectionLoss
from dataLoader import UCFDataLoader, Rescale

class Trainer():
    def __init__(self, init_model = None):
        self.num_classes = 24
        self.num_boxes = 2
        self.image_size = 448

        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        self.dataSet = UCFDataLoader(config['ucf_dataset'],
                                     config['ucf_annotations'],
                                     transform=transforms.Compose([Rescale(self.image_size)]))

        self.net = ActionYolo(num_class=self.num_classes, bbox_num=self.num_boxes).cuda()

        if init_model:
            self.net.load_state_dict(torch.load(init_model))

        self.myDNN = torch.nn.DataParallel(self.net, device_ids=[0, 1, 2])
        self.num_epochs = 1000
        self.batch_size = 16

        self.spatial_channels = 3
        self.temporal_channels = 2

    def train(self):
        dataLoader = DataLoader(self.dataSet, batch_size=1, shuffle=True)
        criterion = DetectionLoss(self.num_classes, self.num_boxes, 7, self.image_size)

        optimizer = torch.optim.Adam(self.myDNN.parameters(), lr=1e-2)

        for epoch in range(self.num_epochs):
            print("Epoch : ", epoch)
            for ex in dataLoader:
                frames = ex['frames'][0]
                label = ex['action'][0]
                start_frame = ex['startFrame'][0] - 1
                end_frame = ex['endFrame'][0] - 1
                action_bbox = ex['bbox'][0]
                frames = frames[start_frame:end_frame + 1]
                flow_images = ex['flowFrames'][0][start_frame:end_frame + 1]

                for b in range(0, len(frames), self.batch_size):

                    print("frames : ", str(b) + "-" + str(b+self.batch_size))

                    batch = frames[b : b + self.batch_size]
                    batch_flow = flow_images[b : b + self.batch_size]

                    batch_flow = Variable(batch_flow.float()).cuda()

                    batch = Variable(batch.float()).cuda()

                    output = self.myDNN.forward(batch, batch_flow)
                    target = action_bbox[b : b + self.batch_size], label

                    loss = criterion(output, target)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                if epoch > 0 and epoch%100 == 0:
                    torch.save(self.net.state_dict(), 'models/' + str(epoch) + '.pth')
