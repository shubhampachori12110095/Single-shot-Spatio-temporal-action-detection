import torch
import torch.nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from utils import Logger



class DetectionLoss(torch.nn.Module):
    """
    Detection loss function.
    """

    def __init__(self, num_classes, num_boxes, grid_size, image_size, iouThresh=0.5, coord_scale = 5, noobj_scale = 1):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.iouThresh = iouThresh
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.grid_size = grid_size
        self.image_size = image_size
        self.cell_size = torch.FloatTensor(np.array([self.image_size / self.grid_size]))


    def forward(self, output, target):
        dtype = torch.FloatTensor

        class_confidence = output[:, :self.num_classes, :, :]
        region_confidence = output[:, self.num_classes : self.num_classes + self.num_boxes, :, :]
        predicted_bbox = output[:, self.num_classes + self.num_boxes:, :, :]

        batch_size = output.data.cpu().numpy().shape[0]

        true_coords, label = target
        tx, ty, tw, th = torch.t(true_coords).float()

        # adjust groundtruth to fit YOLO coordinates format
        tx = tx + (tw / 2.0)
        ty = ty + (th / 2.0)

        cell_x = torch.floor(tx / self.cell_size).long()
        cell_y = torch.floor(ty / self.cell_size).long()

        # Get dimensions relative to Cell coordinates
        tx = tx - (cell_x.float() * self.cell_size)
        ty = ty - (cell_y.float() * self.cell_size)

        I_obj = torch.zeros(batch_size, self.grid_size, self.grid_size)

        # Flip the coordinates (images(column:row))
        I_obj[torch.arange(0, batch_size).long(), cell_x, cell_y] = 1
        I_obj = Variable(I_obj.cuda())

        I_noobj = Variable(torch.ones(batch_size, self.grid_size, self.grid_size).cuda() - I_obj.data)


        cell_boxes = predicted_bbox.contiguous() \
                    .view(-1, self.grid_size, self.grid_size, 8) \
                    [torch.arange(0, batch_size).long().cuda(), cell_x.cuda(), cell_y.cuda()]

        confIoUs, responsible_boxes = self.iou([tx, ty, tw, th], cell_boxes)


        # ToDo: Check this reshape
        px, py, pw, ph = cell_boxes.view(batch_size, 2, 4)[torch.arange(0, batch_size).long().cuda(), torch.LongTensor(responsible_boxes).cuda(), :].t()

        pc = region_confidence[torch.arange(0, batch_size).long().cuda(), torch.LongTensor(responsible_boxes).cuda(), :, :]

        tc = Variable(torch.zeros(self.grid_size, self.grid_size)).cuda()
        tc[cell_x.cuda(), cell_y.cuda()] = dtype(confIoUs).cuda()

        tx = Variable(tx.cuda())
        ty = Variable(ty.cuda())
        tw = Variable(tw.cuda())
        th = Variable(th.cuda())

        pw = F.relu(pw).clamp(max=self.image_size)
        ph = F.relu(ph).clamp(max=self.image_size)

        p_sqrt_w = torch.sqrt(pw)
        p_sqrt_h = torch.sqrt(ph)
        t_sqrt_w = torch.sqrt(tw)
        t_sqrt_h = torch.sqrt(th)

        one_hot_label = torch.zeros(batch_size, self.num_classes, self.grid_size, self.grid_size)
        one_hot_label[:, label - 1] = 1
        one_hot_label = Variable(one_hot_label).cuda()

        coord_loss = self.coord_scale * torch.sum(((tx - px)**2 + (ty - py)**2)) + \
                     self.coord_scale * torch.sum(((t_sqrt_w - p_sqrt_w)**2 + (t_sqrt_h - p_sqrt_h)**2))

        object_loss = torch.sum(I_obj * (tc - pc)**2) + \
                      self.noobj_scale * torch.sum(I_noobj * (tc - pc)**2)

        class_loss = torch.sum(I_obj * torch.sum((F.softmax(class_confidence) - one_hot_label)**2, 1))

        Logger.log_losses([px, py, pw, ph], [tx, ty, tw, th], coord_loss, object_loss, class_loss, confIoUs)

        return coord_loss + object_loss + class_loss

    # ToDo: Make this GPU compatible
    def iou(self, true_bbox, predicted_boxes):

        var2np = lambda x: x.data.cpu().numpy()[0]

        tx, ty, tw, th = true_bbox
        tx1 = tx - (tw / 2.0)
        tx2 = tx + (tw / 2.0)
        ty1 = ty - (th / 2.0)
        ty2 = ty + (th / 2.0)

        ious = []
        responsible_predictions = []

        for sample_id, sample in enumerate(predicted_boxes):

            max_iou = -1
            responsible_prediction = -1

            for idx in range(int(len(sample)/4)):

                px, py, pw, ph = sample[idx*4:idx*4+4]

                px1 = px - (pw / 2.0)
                px2 = px + (pw / 2.0)
                py1 = py - (ph / 2.0)
                py2 = py + (ph / 2.0)

                dx = max(0, min(tx2[sample_id], var2np(px2)) - min(448, max(tx1[sample_id], var2np(px1))))
                dy = max(0, min(ty2[sample_id], var2np(py2)) - min(448, max(ty1[sample_id], var2np(py1))))


                interArea = dx  * dy
                boxAArea = (tx2[sample_id] - tx1[sample_id]) * (ty2[sample_id] - ty1[sample_id])
                boxBArea = (px2 - px1) * (py2 - py1)

                iou = 1.0 * interArea /(boxAArea + var2np(boxBArea) - interArea)

                if iou < 1e-6:
                    iou = 0.0

                if iou > max_iou:
                    max_iou = iou
                    responsible_prediction = idx

            ious.append(max_iou)
            responsible_predictions.append(responsible_prediction)

        return np.array(ious), np.array(responsible_predictions)