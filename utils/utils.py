#!/usr/bin/env python3
import numpy as np
import cv2
import os
import torch
import PIL.Image as pil_image
from torchvision.transforms import functional as F
import math


def rgb2ycbcr(img):
    if type(img) == np.ndarray:
        y = (
            16.0
            + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2])
            / 256.0
        )
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = (
            16.0
            + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :])
            / 256.0
        )
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(img))


def ycbcr2rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256.0 + 408.583 * img[:, :, 2] / 256.0 - 222.921
        g = (
            298.082 * img[:, :, 0] / 256.0
            - 100.291 * img[:, :, 1] / 256.0
            - 208.120 * img[:, :, 2] / 256.0
            + 135.576
        )
        b = 298.082 * img[:, :, 0] / 256.0 + 516.412 * img[:, :, 1] / 256.0 - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256.0 + 408.583 * img[2, :, :] / 256.0 - 222.921
        g = (
            298.082 * img[0, :, :] / 256.0
            - 100.291 * img[1, :, :] / 256.0
            - 208.120 * img[2, :, :] / 256.0
            + 135.576
        )
        b = 298.082 * img[0, :, :] / 256.0 + 516.412 * img[1, :, :] / 256.0 - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception("Unknown Type", type(img))


def image2tensor(image: np.ndarray, range_norm: bool) -> torch.Tensor:
    tensor = F.to_tensor(image)

    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)

    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool):

    if range_norm:
        tensor = tensor.add_(1.0).div_(2.0)

    image = (
        tensor.squeeze_(0)
        .permute(1, 2, 0)
        .mul_(255)
        .clamp_(0, 255)
        .cpu()
        .numpy()
        .astype("uint8")
    )

    return image


def load_image(name):
    image = pil_image.open(name).convert("RGB")
    return image
