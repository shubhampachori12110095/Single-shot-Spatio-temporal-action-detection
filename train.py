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
                                     transform=transforms.Compose([Rescale(self.image_size)]), subsample=True)

        self.net = ActionYolo(num_class=self.num_classes, bbox_num=self.num_boxes).cuda()

        if init_model:
            self.net.load_state_dict(torch.load(init_model))

        self.myDNN = torch.nn.DataParallel(self.net, device_ids=[0, 1, 2])
        self.num_epochs = 1000
        self.batch_size = 4

        self.spatial_channels = 3
        self.temporal_channels = 2

    def train(self):
        dataLoader = DataLoader(self.dataSet, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        criterion = DetectionLoss(self.num_classes, self.num_boxes, 7, self.image_size)

        optimizer = torch.optim.Adam(self.myDNN.parameters(), lr=1e-3)
	#examples_queue = Queue(32)

        for epoch in range(self.num_epochs):
            for ex in dataLoader:
                print("Epoch : ", epoch)
                frames = ex['frames']
                label = ex['action']
                #start_frame = ex['startFrame'] - 1
                #end_frame = ex['endFrame'] - 1
                action_bbox = ex['bbox']
                flow_images = ex['flowFrames']
                #pdb.set_trace()
                permutation = np.random.permutation(np.prod(frames.size()[:2]))
                dtype = torch.LongTensor              
                frames = frames.view(int(np.prod(frames.size()[:2])), self.spatial_channels, 240, 320)[dtype(permutation)]
                flow_images = flow_images.view(int(np.prod(flow_images.size()[:2])), self.temporal_channels, 240, 320)[dtype(permutation)]
                action_bbox = action_bbox.view(int(np.prod(action_bbox.size()[:2])), 4)[dtype(permutation)]
                label = torch.FloatTensor(np.array([item for item in label for _ in range(8)], dtype=float))[dtype(permutation)]

                print(frames.size())
                batch = frames
                batch_flow = flow_images

                batch_flow = Variable(batch_flow.float()).cuda()

                batch = Variable(batch.float()).cuda()

                output = self.myDNN.forward(batch, batch_flow)
                target = action_bbox, label

                loss = criterion(output, target)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            torch.save(self.net.state_dict(), 'models/' + str(epoch) + '.pth')
