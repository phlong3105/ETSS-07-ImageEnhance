import math
import time

import torch
from utils.common import *
from utils.loss_util import *


def model_fn_decorator(loss_fn, device, mode="train"):
    
    def test_model_fn(args, data, model, save_path, compute_metrics):
        # prepare input and forward
        # number     = data["number"]
        cur_psnr   = 0.0
        cur_ssim   = 0.0
        cur_lpips  = 0.0

        in_img     = data["image"].to(device)
        label      = data["ref_image"].to(device)
        b, c, h, w = in_img.size()

        # pad image such that the resolution is a multiple of 32
        w_pad     = (math.ceil(w / 32) * 32 - w) // 2
        h_pad     = (math.ceil(h / 32) * 32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            st = time.time()
            out_1, out_2, out_3 = model(in_img)
            cur_time = time.time() - st
            if h_pad != 0:
               out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
               out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

        if args.TEST["EVALUATION_METRIC"]:
            cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(out_1, label)

        # save images
        # if args.TEST["SAVE_IMG"]:
        #     out_save = out_1.detach().cpu()
        #     torchvision.utils.save_image(out_save, save_path + "/" + "test_%s" % number[0] + ".%s" % args.SAVE_IMG)

        return cur_psnr, cur_ssim, cur_lpips, cur_time

    def model_fn(args, data, model, iters):
        model.train()
        # prepare input and forward
        in_img = data["image"].to(device)
        label  = data["ref_image"].to(device)
        out_1, out_2, out_3 = model(in_img)
        loss   = loss_fn(out_1, out_2, out_3, label)
        psnr   = PSNR()(out_1, label)
        ssim   = SSIM()(out_1, label)
        
        # save images
        if iters % args.TRAIN["SAVE_ITER"] == (args.TRAIN["SAVE_ITER"] - 1):
            in_save     = in_img.detach().cpu()
            out_save    = out_1.detach().cpu()
            gt_save     = label.detach().cpu()
            res_save    = torch.cat((in_save, out_save, gt_save), 3)
            save_number = (iters + 1) // args.TRAIN["SAVE_ITER"]
            # torchvision.utils.save_image(res_save, args.VISUALS_DIR + "/visual_x%04d_" % args.SAVE_ITER + "%05d" % save_number + ".jpg")

        return loss, psnr, ssim

    if mode == "test":
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
