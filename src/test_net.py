import os
import sys
import imageio
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from torch.utils.data import DataLoader
from surface_distance import metrics
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from metrics import single_dice_coefficient 
from datasets import load_dataset
from models import load_model


def test_worker(args):
    if not args.smoke_test:
        sys.stdout = open('{}/test_log.txt'.format(args.save_name), "w")

    kfold = 5 if args.kfold else 1
    hasudorff_sum = 0
    ji_sum = 0
    dsc_sum = 0
    acc_sum = 0
    sens_sum = 0
    dsc_list = []
    hd_list = []
    ji_list = []
    ratio_list = []

    for fold in range(kfold):
        test_dataset = load_dataset(args, fold=fold, train=False)    
        model, batch_size, workers = load_model(args)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=workers, pin_memory=True, drop_last=False)
        # create results folder
        dirname = '{}/results'.format(args.save_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        # load corresponding model
        best_model_path = '{}/model_best_{}.pth.tar'.format(args.save_name, str(fold))
        checkpoint = torch.load(best_model_path)
        state_dict = checkpoint['state_dict']
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.cuda()
        model.eval()

        with torch.no_grad():
            for _, (images, targets_subset, masks_subset, names_subset) in enumerate(test_loader):
                images = images.cuda()
                # compute output
                output = model(images, inference=True)

                for idx, name in enumerate(names_subset):
                    h, w, c = images.size(2), images.size(3), output.size(1)
                    output_np = output.cpu().numpy()[idx, 1]
                    binary_output = np.array(output_np > 0.5, dtype=np.uint8)
                    # write result
                    filename = os.path.join(dirname, str(name) + '.jpg')
                    imageio.imwrite(filename, (binary_output*255).astype(np.uint8))
                    # dsc = single_dice_coefficient(output.cpu()[idx].unsqueeze(0), targets_subset.cpu()[idx].unsqueeze(0))
                    # evaluation
                    target_np = targets_subset.cpu().numpy()[idx].astype(np.uint8)
                    # compute hausdorff distance
                    if binary_output.sum() == 0:
                        binary_output[h//2, w//2] = 1
                    surface_distances = metrics.compute_surface_distances(
                                            target_np.astype(np.bool), binary_output.astype(np.bool), spacing_mm=(1, 1))
                    hasudorff = metrics.compute_robust_hausdorff(surface_distances, 100)
                    hasudorff_sum += hasudorff
                    # compute dsc, sens and acc
                    target_1d = np.reshape(target_np, (-1, 1))
                    pred_1d = np.reshape(binary_output, (-1, 1))
                    accuracy = accuracy_score(target_1d, pred_1d)
                    dsc = f1_score(target_1d, pred_1d)
                    jaccard = jaccard_score(target_1d, pred_1d)
                    pred_mask = target_1d * pred_1d
                    sensitivity = np.sum(pred_mask) / np.sum(target_1d)
                    # area ratio
                    mask_1d = masks_subset.cpu().numpy()[idx].astype(np.uint8)
                    ratio = target_1d.sum() / mask_1d.sum()
                    ratio = target_1d.sum() / (512 * 512)
                    dsc = single_dice_coefficient(output.cpu()[idx].view(1, c, h, w), targets_subset.cpu()[idx].view(1, h, w))
                    print('[{}] Acc: {}, Sens: {}, DSC: {}, HD: {}'.format(name, accuracy, sensitivity, dsc, hasudorff))
                    dsc_sum += dsc
                    ji_sum += jaccard
                    acc_sum += accuracy
                    sens_sum += sensitivity
                    dsc_list.append(dsc)
                    hd_list.append(hasudorff)
                    ji_list.append(jaccard)
                    ratio_list.append(ratio)
                    hd_list.append(hasudorff)

    print('Average DSC: {}'.format(dsc_sum / len(dsc_list)))
    print('DSC Variance: {}'.format(np.std(np.array(dsc_list))))
    print('Average HD: {}'.format(hasudorff_sum / len(dsc_list)))
    print('HD Variance: {}'.format(np.std(np.array(hd_list))))
    print('Average JI: {}'.format(ji_sum / len(dsc_list)))
    print('JI Variance: {}'.format(np.std(np.array(ji_list))))
