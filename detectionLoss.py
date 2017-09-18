import torch
import torch.nn


class DetectionLoss(torch.nn.Module):
    """
    Detection loss function.
    """

    def __init__(self, iouThresh=0.5):
        super(DetectionLoss, self).__init__()
        self.iouThresh = iouThresh

    def forward(self, output, target):
        pass


    def iou(self, bbox1, bbox2):

        x11, y11, w1, h1 = bbox1
        x21, y21, w2, h2 = bbox2

        x12 = x11 + w1
        y12 = y11 - h1
        x22 = x21 + w2
        y22 = y21 - h2

        print(y11, y12, y21, y22)
        dx = min(x12, x22) - max(x11, x21)
        dy = min(y11, y21) - max(y12, y22)

        print(dx, dy)
        if dx < 0 or dy < 0:
            return 0.0

        interArea = dx  * dy
        boxAArea = (x12 - x11) * (y11 - y12)
        boxBArea = (x22 - x21) * (y21 - y22)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou