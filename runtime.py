from train import Trainer

trainer = Trainer(init_model='models/21.pth')
trainer.train()

# from inference import Inference
# from torch.utils.data import DataLoader
# from dataLoader import UCFDataLoader, Rescale
# import yaml
# from torchvision import transforms
# from torch.autograd import Variable
#
#
# with open('config.yaml', 'r') as f:
#     config = yaml.load(f)
#
# dataSet = UCFDataLoader(config['ucf_dataset'],
#                              config['ucf_annotations'],
#                              transform=transforms.Compose([Rescale(448)]), subsample=False)
#
# dataLoader = DataLoader(dataSet, batch_size=1, shuffle=False, pin_memory=True)
#
# sample = []
# for ex in dataLoader:
#     sample.append(ex)
#     break
#
# ex = sample[0]
#
# frames = ex['frames'][0]
# label = ex['action'][0]
# start_frame = ex['startFrame'][0] - 1
# end_frame = ex['endFrame'][0] - 1
# action_bbox = ex['bbox'][0]
# flow_images = ex['flowFrames'][0]
#
# flow_images = Variable(flow_images.float()).cuda()
# frames = Variable(frames.float()).cuda()
#
# predictor = Inference('models/3.pth')
# predictor.detect(frames[:8], flow_images[:8])
