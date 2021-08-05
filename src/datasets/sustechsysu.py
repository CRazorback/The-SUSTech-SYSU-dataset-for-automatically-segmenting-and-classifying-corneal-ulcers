import os
import random
import glob
import torch
import kornia
import cv2

import numpy as np
import kornia.augmentation as K

from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import KFold, train_test_split
from PIL import Image
from tqdm import tqdm
from datasets.data_utils import analyze_name, random_crop_m, stat
from models.positional_encoding import SinusoidalPositionalEmbedding
from scipy.ndimage import distance_transform_edt


class SUSTechSYSU(Dataset):
    def __init__(self, x, y, masks, names, args, train=False):
        assert len(x) == len(y)
        assert len(x) == len(masks)
        assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.masks = masks
        self.names = names
        self.train = train

        # augmentation
        self.crop = not args.no_crop
        self.hflip = K.RandomHorizontalFlip()
        self.vflip = K.RandomVerticalFlip()
        self.jit = K.ColorJitter(0.2, 0.2, 0.05, 0.05)
        self.resize = kornia.geometry.resize
        # self.normalize = K.Normalize(mean=torch.tensor([0.150, 0.404, 0.456]), 
        #                              std=torch.tensor([0.150, 0.239, 0.238]))
        self.normalize = K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), 
                                     std=torch.tensor([0.5, 0.5, 0.5]))
        # inductive bias
        self.inductive_bias = args.inductive_bias

    def __len__(self):
        if self.train:
            return self.dataset_size * 2
        else:
            return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
            return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._get_index(idx)

        # BGR -> RGB -> PIL
        image = self.x[idx][...,::-1].copy()  
        label = self.y[idx].copy()
        mask = self.masks[idx].copy()
        name = self.names[idx]

        if self.inductive_bias == 'CSE':
            encoder = SinusoidalPositionalEmbedding(embedding_dim=4, padding_idx=0, init_size=1024)
            embed_map = encoder(torch.zeros(1, 1, 1024, 1024)).squeeze(0).permute(1, 2, 0)
        elif self.inductive_bias == 'distance':
            embed_map = distance_transform_edt(mask)
            embed_map = (embed_map / np.max(embed_map) - 0.5) / 0.5
            embed_map = torch.tensor(embed_map)
        else:
            embed_map = mask.copy()
            embed_map = torch.tensor(embed_map)

        if self.train and self.crop and random.random() > 0.2:
            image, label, embed_map = random_crop_m(image, label, embed_map, roi=mask, size=[0.5, 0.9])

        image_t = torch.tensor(image / 255).permute(2, 0, 1).unsqueeze(0)
        label_t = torch.tensor(label // 255).unsqueeze(0).float()
        map_t = embed_map.clone().detach().reshape(1, -1, image_t.size(-2), image_t.size(-1)).float()

        if self.train:
            hflip_params = self.hflip.forward_parameters(image_t.shape)
            image_t = self.hflip(image_t, hflip_params)
            label_t = self.hflip(label_t, hflip_params)
            map_t = self.hflip(map_t, hflip_params) 
            vflip_params = self.vflip.forward_parameters(image_t.shape)
            image_t = self.vflip(image_t, vflip_params)
            label_t = self.vflip(label_t, vflip_params)
            map_t = self.vflip(map_t, vflip_params)
            image_t = self.resize(image_t, size=512, interpolation='bilinear', align_corners=False)
            label_t = self.resize(label_t, size=512, interpolation='nearest')
            map_t = self.resize(map_t, size=512, interpolation='nearest')
            jit_params = self.jit.forward_parameters(image_t.shape)
            image_t = self.jit(image_t, jit_params)
        else:
            image_t = self.resize(image_t, size=512, interpolation='bilinear', align_corners=False)
            label_t = self.resize(label_t, size=512, interpolation='nearest')
            map_t = self.resize(map_t, size=512, interpolation='nearest')
            map_t = map_t.view(1, -1, 512, 512)

        image_t = self.normalize(image_t).squeeze(0).float()
        label_t = label_t.long().squeeze()
        map_t = map_t.squeeze(0).float()

        # io debug
        # if not self.train:
        #     import imageio
        #     im_np = image_t.permute(1, 2, 0).numpy()
        #     im_np = (im_np * 0.5 + 0.5) * 255
        #     gt_np = label_t.numpy() * 255
        #     map_np = (map_t.squeeze().numpy() * 0.5 + 0.5) * 255
        #     imageio.imwrite('debug/im.png', im_np.astype(np.uint8))
        #     imageio.imwrite('debug/gt.png', gt_np.astype(np.uint8))
        #     imageio.imwrite('debug/map.png', map_np.astype(np.uint8))

        if self.inductive_bias != '':
            image_t = torch.cat([image_t, map_t], dim=0)

        if self.train:
            return image_t, label_t
        else:
            return image_t, label_t, map_t, name


def load_data(dataset_name):
    input_pattern = dataset_name + '/rawImages/{}.jpg'
    mask_pattern = dataset_name + '/corneaLabels/{}.png'
    targetlist = glob.glob(dataset_name + '/ulcerLabels/*.png')
    targetlist.sort()
    inputs, targets, masks, names = [], [], [], []

    # prepare pairs
    for i in tqdm(range(len(targetlist))):
        targetpath = targetlist[i]
        name = analyze_name(targetpath)
        inputpath = input_pattern.format(str(name))
        maskpath = mask_pattern.format(str(name))
        # raw image                                        
        input = cv2.imread(inputpath)
        # label
        target = cv2.imread(targetpath)
        mask = cv2.imread(maskpath)
        target = 255 - target[...,0]
        mask = 255 - mask[..., 0]

        # padding
        s = max(input.shape[0], input.shape[1])
        padded_input = np.zeros((s, s, 3), np.uint8)
        padded_target = np.zeros((s, s), np.uint8)
        padded_mask = np.zeros((s, s), np.uint8)
        if input.shape[1] > input.shape[0]:
            hu = (input.shape[1] - input.shape[0]) // 2
            hd = hu + input.shape[0]
            padded_input[hu:hd, :, :] = input
            padded_target[hu:hd, :] = target
            padded_mask[hu:hd, :] = mask
        else:
            wl = (input.shape[0] - input.shape[1]) // 2
            wr = wl + input.shape[1]
            padded_input[:, wl:wr, :] = input
            padded_target[:, wl:wr] = target
            padded_mask[:, wl:wr] = mask

        padded_input = cv2.resize(padded_input, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        padded_target = cv2.resize(padded_target, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        padded_mask = cv2.resize(padded_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        inputs.append(padded_input)
        targets.append(padded_target)
        masks.append(padded_mask)
        names.append(name)

    inputs = np.array(inputs)
    targets = np.array(targets)
    masks = np.array(masks)
    names = np.array(names)

    return inputs, targets, masks, names

def load_dataset(args, fold, train=True):
    inputs, targets, masks, names = load_data(args.dataroot)
    # mean & variance
    # mean, std = stat(inputs, masks)
    # normalize = transforms.Normalize(mean=mean.copy(),
    #                                  std=std.copy())
    # print(mean, std)
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=726)
    for ifold, (train_index, test_index) in enumerate(kf.split(targets)):
        if ifold != fold:
            continue
        X_trainset, X_test = inputs[train_index], inputs[test_index]
        y_trainset, y_test = targets[train_index], targets[test_index]
        names_trainset, names_test = names[train_index], names[test_index]
        masks_trainset, masks_test = masks[train_index], masks[test_index]

        X_train, X_val, y_train, y_val, names_train, names_val, masks_train, masks_val = \
            train_test_split(X_trainset, y_trainset, names_trainset, masks_trainset, test_size=0.25, random_state=726)

        train_dataset = SUSTechSYSU(X_train, y_train, masks_train, names_train, args, train=True)
        val_dataset = SUSTechSYSU(X_val, y_val, masks_val, names_val, args, train=False)
        test_dataset = SUSTechSYSU(X_test, y_test, masks_test, names_test, args, train=False)

    if train:
        return train_dataset, val_dataset
    else:
        return test_dataset