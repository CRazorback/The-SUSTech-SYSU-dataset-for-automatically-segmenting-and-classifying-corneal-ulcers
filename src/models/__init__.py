import os 

import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import import_module


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resize=None):
        super().__init__()
        self.resize = resize
        if self.resize == 'up':
            self.f = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.f = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

    def forward(self, x1, x2=None):
        if x2 is not None:
            input = torch.cat([x1, x2], dim=1)
            output = self.f(input)
        else:
            output = self.f(x1)

        return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def init_weights(self, pretrained):
        if os.path.isfile(pretrained):
            print('=> loading pretrained model {}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            pretrained_dict = pretrained_dict['state_dict']
            model_dict = self.state_dict()
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                print('=> available {}'.format(k))
                if k in model_dict.keys():
                    available_pretrained_dict[k] = v
                if k[7:] in model_dict.keys():
                    available_pretrained_dict[k[7:]] = v

            for k, _ in available_pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, pretrained))

            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict)

    def mscale_inference(self, x, scales=[0.5, 1.0, 2.0]):
        x_1x = x

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        scales = sorted(scales, reverse=True)
        pred_avg = 0

        for s in scales:
            x = F.interpolate(x_1x, scale_factor=s, mode='bilinear', 
                                align_corners=False, recompute_scale_factor=True)
            pred = self.forward(x)
            pred = F.interpolate(pred, scale_factor=1/s, mode='bilinear', 
                                align_corners=False, recompute_scale_factor=True)
            pred_avg += pred

        return pred_avg / len(scales)


def load_ddp_model(ngpus_per_node, args):
    print("=> creating model '{}'".format(args.model))
    m = import_module('models.' + args.model.lower())
    
    if args.inductive_bias == 'distance':
        input_channels = 4
    elif args.inductive_bias == 'CSE':
        input_channels = 11
    else:
        input_channels = 3
        
    model = getattr(m, args.model)(input_channels=input_channels)
    model.init_weights(args.pretrained)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = int(args.batch_size / ngpus_per_node)
            workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        batch_size = args.batch_size
        workers = args.workers
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    return model, batch_size, workers


def load_model(args):
    print("=> creating model '{}'".format(args.model))
    m = import_module('models.' + args.model.lower())

    if args.inductive_bias == 'distance':
        input_channels = 4
    elif args.inductive_bias == 'CSE':
        input_channels = 11
    else:
        input_channels = 3

    model = getattr(m, args.model)(input_channels=input_channels)
    model.init_weights(args.pretrained)

    batch_size = args.test_batch_size
    workers = args.workers // args.gpus

    return model, batch_size, workers