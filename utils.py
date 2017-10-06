import cv2
import numpy as np
from torch import nn

class ImageUtils():

    def ComputeOpticalFlow(frames, path=None):
        rgb = np.zeros_like(frames[0])

        frames = ImageUtils.Convert2GrayScale(frames)

        flow = np.empty((len(frames), *frames[0].shape, 3))
        flow[0] = np.zeros_like((*frames[0].shape, 3), dtype=np.float32)

        for idx in range(1, len(frames)):
            snippet = frames[idx - 1:idx + 1]

            frame_flow = np.zeros_like((snippet[0].shape, 2), dtype=np.float32)

            for i in range(len(snippet) - 1):
                frame_flow =  np.add(frame_flow, cv2.calcOpticalFlowFarneback(snippet[i], snippet[i+1], None, 0.5, 3, 15, 3, 21, 2, 0))

            rgb[..., 2] = 0
            rgb[..., 0] = frame_flow[..., 0]
            rgb[..., 1] = frame_flow[..., 1]

            flow[idx] = rgb

        return flow

    def Convert2GrayScale(images):
        grays = []
        for img in images:
            grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        return grays

class Conv2d_BatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyRelu(x)

        return x


class Logger():

    def log_losses(predicted_points, target_points, coord_loss, object_loss, class_loss, confIoUs):
        px, py, pw, ph = predicted_points
        tx, ty, tw, th = target_points
        print("predictions: ", px.data.cpu().numpy()[0], py.data.cpu().numpy()[0], pw.data.cpu().numpy()[0], ph.data.cpu().numpy()[0])
        print("Labels: ", tx.data.cpu().numpy()[0], ty.data.cpu().numpy()[0], tw.data.cpu().numpy()[0], th.data.cpu().numpy()[0])
        print("Coord Loss: ", coord_loss.data.cpu().numpy()[0])
        print("Object Loss: ", object_loss.data.cpu().numpy()[0])
        print("Class Loss: ", class_loss.data.cpu().numpy()[0])
        print(" IoU: ", np.mean(confIoUs))
        print("--------------------------")
