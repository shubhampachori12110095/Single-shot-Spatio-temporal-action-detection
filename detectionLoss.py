import torch
import torch.nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from utils import Logger
import pdb


class DetectionLoss(torch.nn.Module):
    """
    Detection loss function.
    """

    def __init__(self, num_classes, num_boxes, grid_size, image_size, iouThresh=0.5, coord_scale = 1, noobj_scale = 1):
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
        tx = torch.clamp(tx + (tw / 2.0), max=self.image_size - 1)
        ty = torch.clamp(ty + (th / 2.0), max=self.image_size - 1)

        cell_x = torch.floor(tx / self.cell_size).long()
        cell_y = torch.floor(ty / self.cell_size).long()

        # Get dimensions relative to Cell coordinates
        tx = tx - (cell_x.float() * self.cell_size)
        ty = ty - (cell_y.float() * self.cell_size)

        print(predicted_bbox.size())
        cell_boxes = predicted_bbox.contiguous() \
                    .permute(0, 2, 3, 1) \
                    [torch.arange(0, batch_size).long().cuda(), cell_x.cuda(), cell_y.cuda()]

        confIoUs, responsible_boxes = self.iou([tx, ty, tw, th], cell_boxes)

        I_obj = torch.zeros(batch_size, self.grid_size, self.grid_size, self.num_boxes)

        I_obj[torch.arange(0, batch_size).long(), cell_x, cell_y, responsible_boxes] = 1
        I_obj = Variable(I_obj.cuda())

        I_noobj = Variable(torch.ones(batch_size, self.grid_size, self.grid_size, self.num_boxes).cuda() - I_obj.data)

        # ToDo: Check this reshape
        cell_px, cell_py, cell_pw, cell_ph = cell_boxes.view(batch_size, 4, 2).permute(0, 2, 1)[torch.arange(0, batch_size).long().cuda(), torch.LongTensor(responsible_boxes).cuda(), :].t()

        pc = region_confidence #[torch.arange(0, batch_size).long().cuda(), torch.LongTensor(responsible_boxes).cuda(), :, :]

        tc = Variable(torch.zeros(self.grid_size, self.grid_size)).cuda()
        tc[cell_x.cuda(), cell_y.cuda()] = dtype(confIoUs).cuda()

        tx = Variable(tx.cuda())
        ty = Variable(ty.cuda())
        tw = Variable(tw.cuda())
        th = Variable(th.cuda())

        cell_tx, cell_ty, cell_tw, cell_th = tx, ty, tw, th

        px, py, pw, ph = predicted_bbox.permute(1, 2, 3, 0).contiguous().view(4, 2, self.grid_size, self.grid_size, -1)

        px = px.permute(3, 1, 2, 0)
        py = py.permute(3, 1, 2, 0)
        pw = pw.permute(3, 1, 2, 0)
        ph = ph.permute(3, 1, 2, 0)

        # pdb.set_trace()

        tx = torch.mul(I_obj, tx.expand(self.num_boxes, self.grid_size, self.grid_size, batch_size).permute(3, 2, 1, 0))
        ty = torch.mul(I_obj, ty.expand(self.num_boxes, self.grid_size, self.grid_size, batch_size).permute(3, 2, 1, 0))
        tw = torch.mul(I_obj, tw.expand(self.num_boxes, self.grid_size, self.grid_size, batch_size).permute(3, 2, 1, 0))
        th = torch.mul(I_obj, th.expand(self.num_boxes, self.grid_size, self.grid_size, batch_size).permute(3, 2, 1, 0))

        pw = F.relu(pw).clamp(max=self.image_size)
        ph = F.relu(ph).clamp(max=self.image_size)

        p_sqrt_w = torch.sqrt(pw)
        p_sqrt_h = torch.sqrt(ph)
        t_sqrt_w = torch.sqrt(tw)
        t_sqrt_h = torch.sqrt(th)

        one_hot_label = torch.zeros(batch_size, self.num_classes, self.grid_size, self.grid_size)
        one_hot_label[:, label - 1] = 1
        one_hot_label = Variable(one_hot_label).cuda()

        coord_loss = self.coord_scale * torch.sum(I_obj*((tx - px)**2 + (ty - py)**2)) + \
                     self.coord_scale * torch.sum(I_obj*((t_sqrt_w - p_sqrt_w)**2 + (t_sqrt_h - p_sqrt_h)**2))

        object_loss = torch.sum(I_obj * (tc - pc)**2) + \
                      self.noobj_scale * torch.sum(I_noobj * (tc - pc)**2)

        class_loss = torch.sum(I_obj[:, :, :, 0] * torch.sum((F.softmax(class_confidence) - one_hot_label)**2, 1))

        Logger.log_losses([cell_px, cell_py, cell_pw, cell_ph], [cell_tx, cell_ty, cell_tw, cell_th], coord_loss, object_loss, class_loss, confIoUs)

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

            boxes = sample.view(4, 2).t()

            for idx in range(len(boxes)):

                px, py, pw, ph = boxes[idx]

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