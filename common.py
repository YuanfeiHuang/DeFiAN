import random
import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st
import torch
from torchvision import transforms
import math
from PIL import Image
import matplotlib.pyplot as plt

def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw, c = img_in.shape
    tp = scale * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def set_channel(img_in, img_tar, n_channel):

    def _set_channel(img):
        if len(img.shape) == 3:
            h, w, c = img.shape
        else:
            c = 1
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate([img] * n_channel, 2)

        return img / 255

    return _set_channel(img_in), _set_channel(img_tar)

def np2Tensor(img_in, img_tar, rgb_range):
    def _to_tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        torch_tensor = torch.from_numpy(np_transpose).float()
        # torch_tensor.mul_(rgb_range/255)

        return torch_tensor

    return _to_tensor(img_in), _to_tensor(img_tar)


def augment(img_in, img_tar, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return _augment(img_in), _augment(img_tar)