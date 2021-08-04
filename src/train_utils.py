import os
import math
import shutil
import torch
import numpy as np
import torch.optim as optim


def save_checkpoint(state, is_best, fold, savename, epoch, filename='model_checkpoint.pth.tar'):
    dirname = '{}'.format(savename)
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        print('Saving checkpoint {} as the best model...'.format(epoch))
        shutil.copyfile(os.path.join(dirname, filename), '{}/model_best_{}.pth.tar'.format(savename, str(fold)))


def adjust_learning_rate(optimizer, epoch, epochs, lr, cos=True, schedule=None):
    """ Decay the learning rate based on schedule """
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_alpha(epoch, epochs, min_alpha=0.2):
    step = (1 - min_alpha) / epochs
    return 1 - epoch * step


def make_optimizer(target, args):
    ''' make optimizer '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
        kwargs_optimizer['weight_decay'] = args.weight_decay
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon
    else:
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = 0.9
        kwargs_optimizer['eps'] = 0.99
    
    optimizer = optimizer_class(trainable, **kwargs_optimizer)
    return optimizer, args.lr
