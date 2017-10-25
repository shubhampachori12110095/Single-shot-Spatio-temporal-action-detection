from train import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--init")
parser.add_argument("-valid", default=False, action='store_true')
parser.add_argument("--batch_num", default=2, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--fusion", default='AVERAGE')
parser.add_argument("--plot_rate", default=50, type=int)

args = parser.parse_args()

trainer = Trainer(init_model=args.init,
                  use_valid=args.valid,
                  batch_num= args.batch_num,
                  num_workers=args.num_workers,
                  lr=args.lr,
                  fusion=args.fusion,
                  plot_rate=args.plot_rate)

print('---------- Experiment Configurations -------------')
if args.init is not None:
    print("Init Model: ", args.init)
if args.valid is not None:
    print("Valid Dataset: ", args.valid)
if args.batch_num is not None:
    print("Batch Num: ", args.batch_num)
if args.num_workers is not None:
    print("Num Workers: ", args.num_workers)
if args.lr is not None:
    print("Learning Rate: ", args.lr)
if args.fusion is not None:
    print("Fusion Method: ", args.fusion)
if args.plot_rate is not None:
    print("Plot Rate: ", args.plot_rate)
print('----------------------------------------------')

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
# predictor = Inference('models/21Oct/112.pth')
# predictor.detect(frames[:8], flow_images[:8])
