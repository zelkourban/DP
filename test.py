import os.path as osp
import sys
import glob
import os
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
from natsort import natsorted
from models.SRCNN.test import get_model as srcnn
from models.BSRGAN.test import get_model as bsrgan
from models.ESRGAN.test import get_model as esrgan
from models.RealESRGAN.test import get_model as realesrgan
from models.SRGAN.test import get_model as srgan
from models.SWINIR.test import get_model as swinir
from models.EDSR.test import get_model as edsr
from models.VDSR.test import get_model as vdsr
from models.BSRGAN.utils import utils_blindsr as blindsr
from models.BSRGAN.utils.utils_image import (
    uint2single,
    imread_uint,
    single2uint,
    imsave,
)
from utils import (
    load_image,
    image2tensor,
    tensor2image,
    rgb2ycbcr,
    ycbcr2rgb,
)
import math

all_models = [
    "SRCNN",
    "BSRGAN",
    "ESRGAN",
    "REAL-ESRGAN",
    "SRGAN",
    "SWINIR",
    "VDSR",
    "EDSR",
]


class Model:
    def SRCNN(self):
        return srcnn()

    def BSRGAN(self):
        return bsrgan()

    def ESRGAN(self):
        return esrgan()

    def Real_ESRGAN(self):
        return realesrgan()

    def SRGAN(self):
        return srgan()

    def SWINIR(self):
        return swinir()

    def EDSR(self):
        return edsr()

    def VDSR(self):
        return vdsr()

    def __init__(self, name):
        models = {
            "SRCNN": self.SRCNN,
            "BSRGAN": self.BSRGAN,
            "ESRGAN": self.ESRGAN,
            "REAL-ESRGAN": self.Real_ESRGAN,
            "SRGAN": self.SRGAN,
            "SWINIR": self.SWINIR,
            "EDSR": self.EDSR,
            "VDSR": self.VDSR,
        }

        self.name = name
        self.device = torch.device("cpu")
        self.model = models[name]()

    def sr(self, image):

        if self.name in ["SRCNN", "VDSR"]:

            image_height, image_width = image.size
            image = np.asarray(image)
            new_size = (image_height * 4, image_width * 4)
            image = cv2.resize(image, (image_height * 4, image_width * 4))
            ycbcr = rgb2ycbcr(np.array(image))
            image = ycbcr[..., 0]
            image /= 255.0

        image_tensor = (
            image2tensor(image, range_norm=False).to(self.device).unsqueeze_(0)
        )

        with torch.no_grad():
            sr_tensor = self.model(image_tensor.float()).clamp(0.0, 1.0)
        sr_image = tensor2image(sr_tensor, range_norm=False)
        if self.name in ["SRCNN", "VDSR"]:
            sr_image = np.array(
                [sr_image[..., 0], ycbcr[..., 1], ycbcr[..., 2]]
            ).transpose([1, 2, 0])
            sr_image = np.clip(ycbcr2rgb(sr_image), 0.0, 255.0).astype(np.uint8)

        # print(" Done")
        return sr_image


def sr_image(image_path, model, write_path):
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # print(f"Creating SR image for {image_path.split('/')[-1]}...", end="")
    image = load_image(image_path)
    sr_image = model.sr(image)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        f"{write_path}/{image_path.split('/')[-1].split('.')[0]}_sr_{model.name}.png",
        sr_image,
    )


def bsrgan_degrade_pair(image, factor=4, lr_size=64, shuffle_prob=0.1):
    lr_image, hr_image = blindsr.degradation_bsrgan_plus(
        image, sf=factor, shuffle_prob=shuffle_prob, lq_patchsize=lr_size
    )
    return lr_image, hr_image


def downscale_img(image, factor=4):
    image_height, image_width = image.size
    image = np.asarray(image)
    new_size = (image_height // factor, image_width // factor)
    return cv2.resize(image, (image_height // factor, image_width // factor))


def bsrgan_degrade_folder(path):
    if not os.path.exists("LR_BSRGAN"):
        os.makedirs("LR_BSRGAN")

    if not os.path.exists("HR_BSRGAN"):
        os.makedirs("HR_BSRGAN")

    print(f"Degrading images in folder {path}")
    for image_name in tqdm(natsorted(os.listdir(path))):
        # image = cv2.imread(f"{path}/{image_name}")
        image = uint2single(imread_uint(os.path.join(path, image_name)))
        # image = image.astype(np.float32) / 255.0
        lr_image, hr_image = bsrgan_degrade_pair(image, lr_size=128)

        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
        # hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR)
        imsave(single2uint(lr_image), f"LR_BSRGAN/{image_name}")
        imsave(single2uint(hr_image), f"HR_BSRGAN/{image_name}")


def downscale_folder(path):
    if not os.path.exists("LR"):
        os.makedirs("LR")

    for image_name in os.listdir(path):
        image = load_image(f"{path}/{image_name}")
        down_img = downscale_img(image)
        down_image = cv2.cvtColor(down_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"LR/{image_name}", down_img)


def load_model(name):

    if name.upper() in all_models:
        m = Model(name.upper())
        return m
    else:
        raise Exception(f'No model named "{name}", available models {all_models}')


def test_all_models(dataset_path):
    for m in all_models:
        print(f"Processing dataset for {m} model...")
        model = load_model(m)
        test_dataset(dataset_path, model)


def test_dataset(dataset_path, model):

    files = os.listdir(dataset_path)
    files = natsorted(files)
    SR_path = f"SR/{model.name}"
    for f in tqdm(files):

        sr_image(f"{dataset_path}/{f}", model, SR_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset folder")
    parser.add_argument("--model", type=str, help="SR model", default=None)
    parser.add_argument(
        "--mode",
        type=str,
        help="selects mode (available modes: sr_image, sr_folder, sr_all_models, downscale, bsrgan_degrade)",
        default="sr_image",
    )

    parser.add_argument("--all", action="store_true", help="Test on all models")
    parser.add_argument("--input-image", "-i", type=str, help="image name")
    parser.add_argument("--output", type=str, default="SR")

    args = parser.parse_args()

    if args.mode == "sr_image":

        if args.model is None:
            parser.print_help()
            sys.exit(1)
        model = load_model(args.model)
        SR_path = f"{args.output}/{model.name}"
        if args.input_image is not None:

            sr_image(args.input_image, model, SR_path)
    elif args.mode == "sr_folder":

        model = load_model(args.model)
        if args.dataset is not None:
            test_dataset(args.dataset, model)
    elif args.mode == "sr_all_models":
        # if args.downscale is not None:
        test_all_models(args.dataset)
    elif args.mode == "downscale":
        downscale_folder(args.dataset)
    elif args.mode == "bsrgan_degrade":
        bsrgan_degrade_folder(args.dataset)


if __name__ == "__main__":
    main()
