from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):

    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Mini-Batches',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)



class DetectionLossPlotter(object):

    def __init__(self, buffer_size=100):
        self.class_loss = []
        self._object_loss = []
        self.coord_loss = []
        self.iou = []
        self.buffer_size = buffer_size
        self.plotter = VisdomLinePlotter()

    def plot(self, class_loss, object_loss, coord_loss, iou, iteration):
        self.class_loss.append(np.mean(class_loss.data.cpu().numpy()))
        self._object_loss.append(np.mean(object_loss.data.cpu().numpy()))
        self.coord_loss.append(np.mean(coord_loss.data.cpu().numpy()))
        self.iou.append(np.mean(iou))

        if len(self.class_loss) >= self.buffer_size:
            self.plotter.plot("class loss", 'train', iteration, np.mean(np.array(self.class_loss)))
            self.plotter.plot("coord loss", 'train', iteration, np.mean(np.array(self.coord_loss)))
            self.plotter.plot("object loss", 'train', iteration, np.mean(np.array(self._object_loss)))
            self.plotter.plot("Average IoU", 'train', iteration, np.mean(np.array(self.iou)))
            self.clear_buffers()

    def clear_buffers(self):
        self.class_loss = []
        self._object_loss = []
        self.coord_loss = []
        self.iou = []