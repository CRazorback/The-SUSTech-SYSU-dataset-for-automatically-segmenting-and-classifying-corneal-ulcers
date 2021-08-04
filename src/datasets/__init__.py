# -*- coding: utf-8 -*-

from importlib import import_module


def load_dataset(args, fold=0, train=True):
    print("=> creating dataset [{}], fold {}...".format(args.dataset_name, fold))
    m = import_module('datasets.' + args.dataset_name.lower())
    if train:
        # training mode
        train_dataset, val_dataset = m.load_dataset(args, fold, train)
        return train_dataset, val_dataset
    else:
        # testing mode
        test_dataset = m.load_dataset(args, fold, train)
        return test_dataset