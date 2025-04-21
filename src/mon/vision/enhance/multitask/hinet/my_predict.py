#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import config.options as option
import cv2
import data.util as dutil
import numpy as np
import torch
import utils.util as util
from models import create_model

import mon

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = mon.Path(args.save_dir)
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    keep_subdirs = args.keep_subdirs
    opt_path     = str(current_dir / "model_config" / args.opt_path)
    
    # Override options with args
    opt           = option.parse(opt_path, is_train=False)
    opt           = option.dict_to_nonedict(opt)
    opt["device"] = device
    
    # Load model
    opt["path"]["pretrain_model_G"] = str(weights)
    model = create_model(opt)
    
    # Measure efficiency score
    if benchmark:
        flops, params, avg_time = model.compute_efficiency_score()
        console.log(f"FLOPs : {flops:.4f}")
        console.log(f"Params: {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta       = datapoint["meta"]
                image_path = mon.Path(meta["path"])
                image      = dutil.read_img(None, str(image_path))
                # image      = image[:, :, ::-1]
                h, w       = mon.image_size(image)
                # image      = cv2.resize(image, (600, 400))
                image      = mon.resize(image, divisible_by=32)
                image_nf   = cv2.blur(image, (5, 5))
                image_nf   = image_nf * 1.0 / 255.0
                image_nf   = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
                image      = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
                image      = image.unsqueeze(0).to(device)
                image_nf   = image_nf.unsqueeze(0).to(device)
                
                # Infer
                timer.tick()
                model.feed_data(
                    data = {
                        "idx": i,
                        "LQs": image,
                        "nf" : image_nf,
                    },
                    need_GT=False
                )
                model.test()
                timer.tock()
                
                # Post-processing
                visuals        = model.get_current_visuals(need_GT=False)
                enhanced_image = util.tensor2img(visuals["rlt"])  # uint8
                enhanced_image = cv2.resize(enhanced_image, (w, h))
                
                # Save
                if save_image:
                    output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                    cv2.imwrite(str(output_path), enhanced_image)
                    # torchvision.utils.save_image(enhanced_image, str(output_path))
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
    



# ----- Main -----

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
