import os
import sys
import argparse
from distributed import lanuch_mp_worker
from train_net import train_worker
from test_net import test_worker


def main():
    parser = argparse.ArgumentParser(description='Ocular Staining Segmentation')
    # multi-processing
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--gpus', default=4, type=int,
                        help='GPUs to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    # training configuration
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--test_batch_size', default=1, type=int,
                        metavar='N',
                        help='inference mini-batch size (default: 1)')
    parser.add_argument('-e', '--eval_freq', default=5, type=int,
                        metavar='N', help='validation frequency (default: 5)')
    parser.add_argument('--epoch', default=100, type=int,
                        metavar='N', help='training epoch (default: 100)')
    parser.add_argument('--resume', default=-1, type=int, metavar='N',
                        help='resume from which fold (default: -1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--pretrained', default='',
                        help='pretrained model weights')

    # inference configuration
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')

    # model specifications
    parser.add_argument('--model', default='UNet',
                        help='model name')
    parser.add_argument('--inductive_bias', default='',
                        help='(CSE | distance)')
    parser.add_argument('--aux_weight', default=0.0, type=float,
                        help='weight for aux logits')
    parser.add_argument('--multi_scale', action='store_true',
                        help='multi scale inference')
    parser.add_argument('--no_crop', action='store_true',
                        help='disable random resized cropping')

    # experiment configuration
    parser.add_argument('--save_name', default='smoke',
                        help='experiment name')
    parser.add_argument('--dataset_name', default='sustechsysu',
                        help='dataset name')
    parser.add_argument('--dataroot', default='..',
                        help='dataset root path')                   
    parser.add_argument('--kfold', action='store_true',
                        help='5-fold cross-validation')
    parser.add_argument('--smoke_test', action='store_true',
                        help='debug mode')

    args = parser.parse_args()

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    dirname = '{}'.format(args.save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if args.evaluate:
        test_worker(args)
    else:
        lanuch_mp_worker(train_worker, args)
        test_worker(args)

    if not args.evaluate and not args.smoke_test:
        sys.stdout.close()


if __name__ == '__main__':
    main()
