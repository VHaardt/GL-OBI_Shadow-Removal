

import argparse
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from test_scripts.dataloader_old import ISTDDataset
from models.ResNet import CustomResNet101, CustomResNet50
from models.UNet import UNetTranslator_S, UNetTranslator
from datetime import datetime
from utils.metrics import PSNR, RMSE, SSIM
from utils.exposure import exposureRGB, exposureRGB_Tens
from utils.blurring import blur_image_border, dilate_erode_mask
import numpy as np
import random
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import tqdm
import cv2
import json
import requests
from torchvision.transforms import v2

import ipdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the dataset root")
    parser.add_argument("--img_height", type=int, default=480, help="height of the images")
    parser.add_argument("--img_width", type=int, default=480, help="width of the images")

    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="gpu to use for training, -1 for cpu")
    parser.add_argument("--save_images", type=bool, default=False, help="save images during validation")
    
    opt = parser.parse_args()
    
    return opt

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



if __name__ == "__main__":

    opt = parse_args() # parse arguments
    print(opt)
    set_seed(42) # set seed for reproducibility

     # Create checkpoint directory
    checkpoint_dir = os.path.join("RESNET_eval", datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.txt"), "w") as f:
        f.write(str(opt))
    
    # Set device (gpu or cpu)
    device = torch.device(f"cuda:{opt.gpu}" if opt.gpu >= 0 and opt.gpu >= torch.cuda.device_count() - 1 else "cpu")
    print(f"Using device: {device}")

    # Call trained ResNet
    resnet_path = ["/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_B/weights/best_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_B/weights/last_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_B2/weights/best_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_B2/weights/last_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_C/weights/best_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_C/weights/last_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_C2/weights/best_resnet_pretrain.pth",
                   "/home/vhaardt/Desktop/ShadowRemoval/code/Shadow-Removal-1/checkpoints/Esp_C2/weights/last_resnet_pretrain.pth"
                   ]
    
    c = 0
    
    for p in resnet_path:
        c +=1
        resnet = CustomResNet50().to(device)
        resnet.load_state_dict(torch.load(p))
        resnet.eval()
        for param in resnet.parameters():
            param.requires_grad = False
    
        # Define loss
        #criterion_pixelwise = torch.nn.MSELoss().to(device)
        criterion_pixelwise = torch.nn.L1Loss().to(device)
        criterion_perceptual = LPIPS(net_type="vgg" , normalize= True).to(device)
        criterion_segm = torch.nn.BCEWithLogitsLoss().to(device)


        # Define metrics
        PSNR_ = PSNR()
        RMSE_ = RMSE()
        SSIM_ = SSIM()

        Tensor = torch.cuda.FloatTensor if opt.gpu >= 0 else torch.FloatTensor

        # Train and test on the same data
        if "ISTD" in opt.dataset_path:
            #train_dataset = ISTDDataset(os.path.join(opt.dataset_path, "train"), size=(opt.img_height, opt.img_width), aug=True, fix_color=True)
            val_dataset = ISTDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True)

        g = torch.Generator()
        g.manual_seed(42)
        
        
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=opt.n_workers,
                                                        worker_init_fn=seed_worker,
                                                        generator=g)
        
        # Init losses and metrics lists
        #p: for partial -> loss of resnet
        #g: for global -> loss of unet

        rmse = []
        psnr = []
        ssim = []
        psnr_shadow = []
        rmse_shadow = []
        ssim_shadow = []

        rmse_epoch = 0
        psnr_epoch = 0
        psnr_epoch_shadow = 0
        rmse_epoch_shadow = 0
        ssim_epoch = 0
        ssim_epoch_shadow = 0

        pbar = tqdm.tqdm(total=val_dataset.__len__())
        for i, data in enumerate(val_loader):
            shadow = data['shadow_image']
            shadow_free = data['shadow_free_image']
            mask = data['shadow_mask']
            crop_coordinate = data['crop_coordinate']

            inp = shadow.type(Tensor).to(device)
            gt = shadow_free.type(Tensor).to(device)
            mask = mask.type(Tensor).to(device)
            crop_coordinate = crop_coordinate.type(Tensor).to(device)

            inp = torch.clamp(inp, 0, 1)
            gt = torch.clamp(gt, 0, 1)

            out_p = resnet(inp)

            transformed_images = exposureRGB(inp, out_p)

            mask_exp = mask.expand(-1, 3, -1, -1)

            ##
            R_a_mat = torch.where(mask == 0., torch.tensor(1.0), out_p[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, inp.size(2), inp.size(3)))
            R_b_mat = torch.where(mask == 0., torch.tensor(0.0), out_p[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, inp.size(2), inp.size(3)))
            G_a_mat = torch.where(mask == 0., torch.tensor(1.0), out_p[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, inp.size(2), inp.size(3)))
            G_b_mat = torch.where(mask == 0., torch.tensor(0.0), out_p[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, inp.size(2), inp.size(3)))
            B_a_mat = torch.where(mask == 0., torch.tensor(1.0), out_p[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, inp.size(2), inp.size(3)))
            B_b_mat = torch.where(mask == 0., torch.tensor(0.0), out_p[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, inp.size(2), inp.size(3)))
            inp_g = torch.cat((R_a_mat, R_b_mat, G_a_mat, G_b_mat, B_a_mat, B_b_mat), dim=1)
            innested_img = exposureRGB_Tens(inp, inp_g)
                        
            dilated_mask, eroded_mask = dilate_erode_mask(mask, kernel_size=10)
            border_mask1 = mask_exp - eroded_mask
            border_mask2 = dilated_mask - mask_exp
            border_mask = border_mask1 + border_mask2
            blurred_innested_img = blur_image_border(innested_img, border_mask, blur_kernel_size=11)

            psnr_ = PSNR_(blurred_innested_img, gt)
            rmse_ = RMSE_(blurred_innested_img, gt)
            ssim_ = SSIM_(blurred_innested_img, gt)

            psnr_shadow_ = PSNR_(blurred_innested_img*mask_exp, gt*mask_exp)
            rmse_shadow_ = RMSE_(blurred_innested_img*mask_exp, gt*mask_exp)
            ssim_shadow_ = SSIM_(blurred_innested_img*mask_exp, gt*mask_exp)

            psnr_epoch += psnr_.detach().item()
            psnr_epoch_shadow += psnr_shadow_.detach().item()
            rmse_epoch += rmse_.detach().item()
            rmse_epoch_shadow += rmse_shadow_.detach().item()
            ssim_epoch += ssim_.detach().item()
            ssim_epoch_shadow += ssim_shadow_.detach().item()
            pbar.update(opt.batch_size)
        pbar.close()

        rmse.append(rmse_epoch / len(val_loader))
        rmse_shadow.append(rmse_epoch_shadow / len(val_loader))
        psnr.append(psnr_epoch / len(val_loader))
        psnr_shadow.append(psnr_epoch_shadow / len(val_loader))
        ssim.append(ssim_epoch / len(val_loader))
        ssim_shadow.append(ssim_epoch_shadow / len(val_loader))

        print(f"[RMSE: {rmse}, RMSE_shadow: {rmse_shadow}, PSNR: {psnr}, PSNR_shadow: {psnr_shadow}, SSIM: {ssim}, SSIM_shadow: {ssim_shadow}]")

        
        # Save metrics to disk
        metrics_dict = {
            "all_RMSE": rmse,
            "shadow_RMSE": rmse_shadow,
            "all_PSNR": psnr,
            "shadow_PSNR": psnr_shadow,
            "all_SSIM": ssim,
            "shadow_ssim": ssim_shadow
        }

        with open(os.path.join(checkpoint_dir, f"{c}_metrics.json"), "w") as f:
            json.dump(metrics_dict, f)