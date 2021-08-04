import os
import random
import cv2
import imageio
import torch
import numpy as np

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

from skimage.io import imsave
from PIL import Image, ImageOps
from pathlib import Path
from multiprocessing.pool import Pool
from torch import einsum
from torch import Tensor
from tqdm import tqdm
from functools import partial


def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def tensor2image(tensor, name):
    tensor_cpu = tensor.byte().permute(1, 2, 0).cpu()
    imageio.imwrite(name, tensor_cpu.numpy())


def crop(img, gt, roi=None, patch_size=128):
    ''' Crop patches with given size '''
    ih, iw = img.shape[:2]
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    if roi is not None:
        # crop a patch in the region of interest
        while roi[iy, ix] == 0:
            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)    

    cropped_img = img[iy:iy+ip, ix:ix+ip]
    cropped_gt = gt[iy:iy+ip, ix:ix+ip]

    return cropped_img, cropped_gt


def crop2(img, gt, roi, feature, patch_size=128):
    ''' Crop patches according to feature maps with given size '''
    ih, iw = feature.size()[1:]
    ip = patch_size // 16

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    # crop a patch in the region of interest
    roi = cv2.resize(roi, (ih, iw), interpolation=cv2.INTER_NEAREST)
    while roi[iy, ix] == 0:
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)    

    cropped_feature = feature[:, iy:iy+ip, ix:ix+ip]
    ix *= 16
    iy *= 16
    ip *= 16
    cropped_img = img[iy:iy+ip, ix:ix+ip]
    cropped_gt = gt[iy:iy+ip, ix:ix+ip]

    return cropped_img, cropped_gt, cropped_feature


def random_crop(img, gt, roi, size=[0.2, 0.8]):
    ''' Crop patches in ROI with random size '''
    import random
    ih, iw = img.shape[:2]
    ip = random.randrange(int(ih * size[0]), int(ih * size[1]))
    ip_l = ip // 2
    ip_r = ip - ip_l

    ix = random.randrange(ip_l, iw - ip_r + 1)
    iy = random.randrange(ip_l, iw - ip_r + 1)
    if roi is not None:
        # crop a patch in the region of interest
        while roi[iy, ix] == 0:
            ix = random.randrange(ip_l, iw - ip_r + 1)
            iy = random.randrange(ip_l, iw - ip_r + 1)    

    cropped_img = img[iy-ip_l:iy+ip_r, ix-ip_l:ix+ip_r]
    cropped_gt = gt[iy-ip_l:iy+ip_r, ix-ip_l:ix+ip_r]

    return cropped_img, cropped_gt


def random_crop_m(img, gt, gt2, roi=None, size=[0.2, 0.8]):
    ''' Crop patches in ROI with random size '''
    import random
    ih, iw = img.shape[:2]
    ip = random.randrange(int(ih * size[0]), int(ih * size[1]))
    ip_l = ip // 2
    ip_r = ip - ip_l

    ix = random.randrange(ip_l, iw - ip_r + 1)
    iy = random.randrange(ip_l, iw - ip_r + 1)
    if roi is not None:
        # crop a patch in the region of interest
        while roi[iy, ix] == 0:
            ix = random.randrange(ip_l, iw - ip_r + 1)
            iy = random.randrange(ip_l, iw - ip_r + 1)    

    l = iy - ip_l
    r = iy + ip_r
    u = ix - ip_l
    d = ix + ip_r
    cropped_img = img[l:r, u:d]
    cropped_gt = gt[l:r, u:d]
    cropped_gt2 = gt2[l:r, u:d]

    return cropped_img, cropped_gt, cropped_gt2


# functions redefinitions
tqdm_ = partial(tqdm, ncols=175,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", a).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", flotted, ws) / tot
    ch = einsum("bcwh,wh->bc", flotted, hs) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)

    return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg.cpu().numpy())


def augment(*arrs: Union[np.ndarray, Image.Image]) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if random() > 0.5:
        angle = random() * 90 - 45
        imgs = map_(lambda e: e.rotate(angle), imgs)
    return imgs


def augment_arr(*arrs_a: np.ndarray) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if random() > 0.5:
        arrs = map_(np.flip, arrs)
    if random() > 0.5:
        arrs = map_(np.fliplr, arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    def g_center(arr):
        if arr.shape == shape:
            return arr

        dx = (arr.shape[0] - shape[0]) // 2
        dy = (arr.shape[1] - shape[1]) // 2

        if dx == 0 or dy == 0:
            return arr[:shape[0], :shape[1]]

        res = arr[dx:-dx, dy:-dy][:shape[0], :shape[1]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, dx, dy)

        return res

    return [g_center(arr) for arr in arrs]


def stat(inputs, masks):
    # RGB mean & std in ROI
    zeros = np.array([0, 0, 0])
    inputs_ = np.reshape(inputs, (-1, 3))
    masks_ = np.reshape(masks, (-1, 1)).squeeze()
    flatten_masks = np.stack([masks_, masks_, masks_], axis=1)
    masked_inputs = inputs_[np.where(np.all(flatten_masks != zeros, axis=-1))]
    mean = np.mean(masked_inputs, axis=0)[::-1] / 255
    std = np.std(masked_inputs, axis=0)[::-1] / 255

    return mean, std