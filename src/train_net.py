# -*- coding: utf-8 -*-
import os
import sys
import time
import builtins
import torch
import imageio

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchmetrics import F1
from torch.utils.data import DataLoader
from log import AverageMeter, ProgressMeter
from train_utils import adjust_learning_rate, save_checkpoint, make_optimizer
from metrics import single_dice_coefficient
from datasets import load_dataset
from models import load_ddp_model
from losses import load_criteria


def train(train_loader, model, criteria, optimizer, epoch, args, weight_dict):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_list = []
    for loss_name in weight_dict:
        loss_list.append(AverageMeter(loss_name, ':.4e'))
    meters = [batch_time, data_time]
    meters.extend(loss_list)

    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(images)
        # combining loss
        loss = 0
        
        if isinstance(outputs, (tuple, list)):
            if weight_dict is None:
                raise NotImplementedError('Expecting a weight dictionary')
            for j, (loss_name, output) in enumerate(zip(weight_dict, outputs)):
                sub_loss = criteria[j](output, target) * weight_dict[loss_name]
                loss += sub_loss
                # record loss
                loss_list[j].update(sub_loss.item(), images.size(0))
        else:
            loss = criteria[0](outputs, target)
            # record loss
            loss_list[0].update(loss.item(), images.size(0))

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        # clip gradient
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, mode='Valid'):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_list = AverageMeter('Loss', ':.4e')
    dscs = AverageMeter('Dice', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, loss_list, dscs],
        prefix='{}: '.format(mode))

    f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')

    # switch to evaluate mode
    model.eval()
    # save groundtruths and predictions
    gt, pred = [], []

    with torch.no_grad():
        end = time.time()
        for i, (images, target, _, _) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images, inference=True)
            # output, images, target = all_gather([output, images, target])
            loss = criterion(output, target)
            # compute metrics
            dsc = f1_score(F.softmax(output, dim=1), target)[1]

            # record loss
            loss_list.update(loss.item(), images.size(0))
            dscs.update(dsc.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 5 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Loss {loss.avg:.5f} * Dice {dice.avg:.5f}'.format(loss=loss_list, dice=dscs))

    return dscs.avg, gt, pred


def train_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        if not args.evaluate and not args.smoke_test:
            sys.stdout = open('{}/train_log.txt'.format(args.save_name), "w")

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    train_kfold_model(gpu, ngpus_per_node, args)


def train_kfold_model(gpu, ngpus_per_node, args):
    kfold = 5 if args.kfold else 1

    for fold in range(kfold):
        train_dataset, val_dataset = load_dataset(args, fold=fold)    
        model, batch_size, workers = load_ddp_model(ngpus_per_node, args)
        criteria = load_criteria(args)
        optimizer, lr = make_optimizer(model, args)

        args.start_epoch = 0
        epochs = args.epoch
        best_dsc = -1e4
        weight_dict = {}

        # contruct weight dictionary if necessary
        if args.aux_weight > 0:
            weight_dict = {'Loss': 1.0, 'Aux Loss': args.aux_weight}
        else:
            weight_dict = {'Loss': 1.0}

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
                                  num_workers=workers, pin_memory=True, drop_last=True, 
                                  sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, 
                                num_workers=workers, pin_memory=True, drop_last=False)

        # optionally resume from a checkpoint
        if args.resume == fold:
            model_path = '{}/model_checkpoint.pth.tar'.format(args.save_name)
            best_model_path = '{}/model_best_{}.pth.tar'.format(args.save_name, str(fold))
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if args.gpu is None:
                    checkpoint = torch.load(model_path)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(model_path, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(model_path, checkpoint['epoch']))
                # load the best metric
                if os.path.isfile(model_path):
                    checkpoint = torch.load(best_model_path)
                    best_dsc = checkpoint['best_metric']
            else:
                print("=> no checkpoint found at '{}'".format(model_path))

        if args.resume <= fold:
            for epoch in range(args.start_epoch, epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                adjust_learning_rate(optimizer, epoch, epochs, lr, cos=True)

                # train for one epoch
                train(train_loader, model, criteria, optimizer, epoch, args, weight_dict)

                # evaluate on validation set
                if epoch % args.eval_freq == 0:
                    dsc, _, _ = validate(val_loader, model, criteria[0], args)

                # remember best dsc and save checkpoint
                is_best = dsc > best_dsc
                best_dsc = max(dsc, best_dsc)

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_metric': best_dsc,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, fold, args.save_name, epoch)

