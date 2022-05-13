#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# from utils_metrics import calculate_psnr  # , calculate_ssim

# from sewar.full_ref import ssim as calculate_ssim
# from sewar.full_ref import psnrb as calculate_psnr
# from prettytable import PrettyTable
from texttable import Texttable
import argparse
from niqe import niqe
import lpips
import torch
from test import test_dataset
import latextable
from sewar.full_ref import ssim as calculate_ssim
from sewar.full_ref import psnr as calculate_psnr
from natsort import natsorted


def mean(l):
    return sum(l) / len(l)


def main(SR_folder, HR_folder, subset=None):
    t = Texttable()
    idx = 0

    if subset != None:
        hr_images = os.listdir(HR_folder)[:subset]
    else:
        hr_images = os.listdir(HR_folder)
    hr_images = natsorted(hr_images)
    t.set_cols_dtype(["t", "f", "f", "f", "f"])
    t.set_deco(Texttable.HEADER)
    t.header(["Image", "PSNR", "SSIM", "LPIPS", "NIQE"])
    lpips_model = lpips.LPIPS(net="alex")
    # if not os.listdir(SR_img_folder):
    #    test_dataset(LR_img_folder)

    psnrs, ssims, lpipss, niqes = [], [], [], []
    for model in tqdm(os.listdir(SR_folder)):

        psnr_list, ssim_list, lpips_list, niqe_list = [], [], [], []
        sr_images = os.listdir(f"{SR_folder}/{model}")
        sr_images = natsorted(sr_images)
        for i in tqdm(range(len(hr_images))):
            idx += 1
            SR_img = cv2.imread(
                # f"{SR_folder}/{model}/{hr_images[i].split('.')[0]}x4_sr_{model}.png",
                f"{SR_folder}/{model}/{sr_images[i]}",
                cv2.IMREAD_COLOR,
            )

            HR_img = cv2.imread(f"{HR_folder}/{hr_images[i]}", cv2.IMREAD_COLOR)
            ssim = calculate_ssim(HR_img, SR_img)[0]
            # SR_img = SR_img * 1.0 / 255

            # HR_img = HR_img * 1.0 / 255

            psnr = calculate_psnr(HR_img, SR_img)

            SR_img_tensor = torch.from_numpy(
                np.transpose(SR_img[:, :, [2, 1, 0]], (2, 0, 1))
            ).float()
            HR_img_tensor = torch.from_numpy(
                np.transpose(HR_img[:, :, [2, 1, 0]], (2, 0, 1))
            ).float()
            lpips_res = (
                lpips_model(SR_img_tensor, HR_img_tensor).data.squeeze().float().numpy()
            )
            SR_img_f32 = np.float32(SR_img)
            niqe_res = niqe(cv2.cvtColor(SR_img_f32, cv2.COLOR_RGB2GRAY))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips_res)
            niqe_list.append(niqe_res)
            t.add_row([hr_images[i].split(".")[0], psnr, ssim, lpips_res, niqe_res])
        print(t.draw())

        psnrs.append(psnr_list)
        ssims.append(ssim_list)
        lpipss.append(lpips_list)
        niqes.append(niqe_list)
        print(f"Processing {model} metrics...")

    t2 = Texttable()
    t2.header(["Model", "PSNR", "SSIM", "LPIPS", "NIQE"])

    t2.set_deco(Texttable.HEADER)
    for i in range(len(psnrs)):
        t2.add_row(
            [
                os.listdir(SR_folder)[i],
                mean(psnrs[i]),
                mean(ssims[i]),
                mean(lpipss[i]),
                mean(niqes[i]),
            ]
        )
    print(t2.draw())
    print(latextable.draw_latex(t2, caption="", label="table:set14_metrics"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", type=str, help="SR folder", required=True)
    parser.add_argument("-hr", type=str, help="HR folder", required=True)
    parser.add_argument("--subset", type=int, help="Subset from dataset")
    args = parser.parse_args()

    if args.subset != None:
        main(args.sr, args.hr, args.subset)
    else:
        main(args.sr, args.hr)
