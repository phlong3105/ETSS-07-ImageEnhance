import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.data import *
from data.options import option
from data.scheduler import *
from eval import eval
from loss.losses import *
from measure import metrics
from net.cidnet import CIDNet

opt = option().parse_args()


def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

  
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

 
def train(epoch):
    model.train()
    loss_print   = 0
    pic_cnt      = 0
    loss_last_10 = 0
    pic_last_10  = 0
    train_len    = len(training_data_loader)
    iter         = 0
    torch.autograd.set_detect_anomaly(True)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1        = im1.cuda()
        im2        = im2.cuda()
        output_rgb = model(im1)  
        gt_rgb     = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi     = model.HVIT(gt_rgb)
        loss_hvi   = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb   = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss       = loss_rgb + opt.HVI_weight * loss_hvi
        iter      += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print    = loss_print + loss.item()
        loss_last_10  = loss_last_10 + loss.item()
        pic_cnt      += 1
        pic_last_10  += 1
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch, loss_last_10/pic_last_10, optimizer.param_groups[0]["lr"]))
            loss_last_10 = 0
            pic_last_10  = 0
            output_img   = transforms.ToPILImage()(output_rgb[0].squeeze(0))
            gt_img       = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
            if not os.path.exists(opt.val_folder + "training"):
                os.mkdir(opt.val_folder + "training")
            output_img.save(opt.val_folder + "training/test.png")
            gt_img.save(opt.val_folder + "training/gt.png")
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


def load_datasets():
    print("===> Loading datasets")
    if opt.lol_v1 or opt.lol_blur or opt.lol_v2_real or opt.lol_v2_synthetic or opt.sid or opt.sice_mix or opt.sice_grad:
        if opt.lol_v1:
            train_set            = get_lol_v1_training_set(opt.data_train_lol_v1, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        elif opt.lol_blur:
            train_set            = get_training_set_blur(opt.data_train_lol_blur, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        elif opt.lol_v2_real:
            train_set            = get_lol_v2_training_set(opt.data_train_lol_v2_real, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_eval_set(opt.data_val_lol_v2_real)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        elif opt.lol_v2_synthetic:
            train_set            = get_lol_v2_synthetic_training_set(opt.data_train_lol_v2_synthetic, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_eval_set(opt.data_val_lol_v2_synthetic)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        elif opt.sid:
            train_set            = get_sid_training_set(opt.data_train_sid, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_eval_set(opt.data_val_sid)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        elif opt.sice_mix:
            train_set            = get_sice_training_set(opt.data_train_sice, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_sice_eval_set(opt.data_val_sice_mix)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        elif opt.sice_grad:
            train_set            = get_sice_training_set(opt.data_train_sice, size=opt.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
            test_set             = get_sice_eval_set(opt.data_val_sice_grad)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
        else:
            raise Exception("Should choose a dataset")
    else:
        raise Exception("Should choose a dataset")
    return training_data_loader, testing_data_loader


def build_model():
    print("===> Building model ")
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model


def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer       = optimizer,
                periods         = [(opt.epochs // 4) - opt.warmup_epochs, (opt.epochs * 3) // 4],
                restart_weights = [1, 1],
                eta_mins        = [0.0002, 0.0000001]
            )
            scheduler = GradualWarmupScheduler(
                optimizer       = optimizer,
                multiplier      = 1,
                total_epoch     = opt.warmup_epochs,
                after_scheduler = scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer       = optimizer,
                periods         = [opt.epochs // 4, (opt.epochs * 3) // 4],
                restart_weights = [1, 1],
                eta_mins        = [0.0002, 0.0000001]
            )
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer       = optimizer,
                periods         = [opt.epochs - opt.warmup_epochs - opt.start_epoch],
                restart_weights = [1],
                eta_min         = 1e-7
            )
            scheduler = GradualWarmupScheduler(
                optimizer       = optimizer,
                multiplier      = 1,
                total_epoch     = opt.warmup_epochs,
                after_scheduler = scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer       =  optimizer,
                periods         = [opt.epochs - opt.start_epoch],
                restart_weights = [1],
                eta_min         = 1e-7
            )
    else:
        raise Exception("should choose a scheduler")
    return optimizer, scheduler


def init_loss():
    L1_weight = opt.L1_weight
    D_weight  = opt.D_weight
    E_weight  = opt.E_weight
    P_weight  = 1.0
    
    L1_loss = L1Loss(loss_weight=L1_weight, reduction="mean").cuda()
    D_loss  = SSIM(weight=D_weight).cuda()
    E_loss  = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss  = PerceptualLoss(
        layer_weights     = {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
        perceptual_weight = P_weight,
        criterion         = "mse"
    ).cuda()
    return L1_loss, P_loss, E_loss, D_loss


if __name__ == "__main__":
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer, scheduler = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss = init_loss()
    
    '''
    train
    '''
    psnr        = []
    ssim        = []
    lpips       = []
    start_epoch = 0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch + 1, opt.epochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size      = True
            
            # LOL three subsets
            if opt.lol_v1:
                output_folder = "LOLv1/"
                label_dir     = opt.data_valgt_lol_v1
            elif opt.lol_v2_real:
                output_folder = "LOLv2_real/"
                label_dir     = opt.data_valgt_lol_v2_real
            elif opt.lol_v2_synthetic:
                output_folder = "LOLv2_syn/"
                label_dir     = opt.data_valgt_lol_v2_synthetic
            # LOL-blur dataset with low_blur and high_sharp_scaled
            elif opt.lol_blur:
                output_folder = "LOL_blur/"
                label_dir     = opt.data_valgt_lol_blur
            elif opt.sid:
                output_folder = "sid/"
                label_dir     = opt.data_valgt_sid
                npy = True
            elif opt.sice_mix:
                output_folder = "sice_mix/"
                label_dir     = opt.data_valgt_sice_mix
                norm_size     = False
            elif opt.sice_grad:
                output_folder = "sice_grad/"
                label_dir     = opt.data_valgt_sice_grad
                norm_size     = False
            else:
                raise ValueError
            
            im_dir = opt.val_folder + output_folder + "*.png"
            eval(
                model               = model,
                testing_data_loader = testing_data_loader,
                model_path          = model_out_path,
                output_folder       = opt.val_folder + output_folder,
                norm_size           = norm_size,
                lol_v1              = opt.lol_v1,
                lol_v2              = opt.lol_v2_real,
                alpha               = 0.8
            )
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_gt_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()
    
    with open("./results/training/metrics.md", "w") as f:
        f.write("dataset: " + output_folder + "\n")
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batch_size}\n")  
        f.write(f"crop size:  {opt.crop_size}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight:  {opt.L1_weight}\n")
        f.write(f"D_weight:   {opt.D_weight}\n")
        f.write(f"E_weight:   {opt.E_weight}\n")
        f.write(f"P_weight:   {opt.P_weight}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {(i + 1) * 10} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")
