

import argparse
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from dataloader import ISTDDataset, SRDDataset, WSRDDataset
from models.ResNet import CustomResNet101, CustomResNet50
from models.UNet import UNetTranslator_S, UNetTranslator
from datetime import datetime
from utils.metrics import PSNR, RMSE
from skimage.metrics import structural_similarity as SSIM_
from utils.exposure import exposureRGB_Tens, exposure_3ch
from utils.blurring import penumbra
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

    parser.add_argument("--resnet_size", type=str, default="S", help="size of the model (S, M)")
    parser.add_argument("--unet_size", type=str, default="S", help="size of the model (S, M)")

    parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="gpu to use for training, -1 for cpu")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--decay_epoch", type=int, default=400, help="epoch from which to start lr decay")
    parser.add_argument("--decay_steps", type=int, default=4, help="number of step decays")

    parser.add_argument("--pixel_weight", type=float, default=1, help="weight of the pixelwise loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.2, help="weight of the perceptual loss")

    parser.add_argument("--valid_checkpoint", type=int, default=1, help="number of epochs between each validation")
    parser.add_argument("--checkpoint_mode", type=str, default="b/l", help="mode for saving checkpoints: b/l (best/last), all (all epochs), n (none)")
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#Delete this in the final version
def send_telegram_notification(message):
    bot_token = '7363314579:AAF5x5LkQrKTk7zJjHh-s5SKUnOWMtitVxs'
    chat_id = '5757693999'
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, json=payload, verify=False)
    return response.json()


if __name__ == "__main__":

    opt = parse_args() # parse arguments
    print(opt)
    set_seed(42) # set seed for reproducibility
    
    # Set device (gpu or cpu)
    device = torch.device(f"cuda:{opt.gpu}" if opt.gpu >= 0 and opt.gpu >= torch.cuda.device_count() - 1 else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.txt"), "w") as f:
        f.write(str(opt))

    # Define model
    if opt.unet_size == "S":
        unet = UNetTranslator_S(in_channels=5, out_channels=6, deconv=False, local=0).to(device) #residual settato False
    else:
        unet = UNetTranslator(in_channels=5, out_channels=6, deconv=False, local=0).to(device)
    unet.apply(weights_init_normal)
    
    n_params1 = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Model has {n_params1} trainable parameters")
    with open(os.path.join(checkpoint_dir, "architecture.txt"), "w") as f:
        f.write("Model has " + str(n_params1) + " trainable parameters\n")
        f.write(str(unet))
    
    # Define loss
    mse_loss = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss().to(device)
    criterion_perceptual = LPIPS(net_type="vgg" , normalize= True).to(device)
    criterion_segm = torch.nn.BCEWithLogitsLoss().to(device)


    # Define metrics
    PSNR_ = PSNR()
    RMSE_ = RMSE()

    # Define optimizer for UNet
    optimizer = torch.optim.Adam(unet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    decay_step = (opt.n_epochs - opt.decay_epoch) // opt.decay_steps
    milestones = [opt.decay_epoch + i * opt.decay_steps for i in range((opt.n_epochs - opt.decay_epoch) // opt.decay_steps)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    Tensor = torch.cuda.FloatTensor if opt.gpu >= 0 else torch.FloatTensor

    # Train and test on the same data
    if "ISTD" in opt.dataset_path:
        train_dataset = ISTDDataset(os.path.join(opt.dataset_path, "train"), size=(opt.img_height, opt.img_width), aug=True, fix_color=True)
        val_dataset = ISTDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True)
    elif "SRD" in opt.dataset_path:
        train_dataset = SRDDataset(os.path.join(opt.dataset_path, "train"), size=(opt.img_height, opt.img_width), aug=False, fix_color=True, masks_precomp=True)
        val_dataset = SRDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=False, masks_precomp=True)
    elif "WRSD" in opt.dataset_path:
        train_dataset = WSRDDataset(os.path.join(opt.dataset_path, "train"), size=(opt.img_height, opt.img_width), aug=True, masks_precomp=False)
        val_dataset = WSRDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True, masks_precomp=False)


    g = torch.Generator()
    g.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True,
                                                  num_workers=opt.n_workers,
                                                  worker_init_fn=seed_worker,
                                                  generator=g)
    
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=opt.n_workers,
                                                    worker_init_fn=seed_worker,
                                                    generator=g)
    
    # Init losses and metrics lists
    #p: for partial -> loss of resnet
    #g: for global -> loss of unet
    
    train_loss = []
    val_loss = []

    val_rmse = []
    val_psnr = []
    val_mae = []
    val_ssim = []
    val_lpips = []

    val_psnr_s = []
    val_rmse_s = []
    val_mae_s = []
    val_ssim_s = []

    val_psnr_ns = []
    val_rmse_ns = []
    val_mae_ns = []
    val_ssim_ns = []

    best_loss = 1e3 # arbitrary large number


    # =================================================================================== #
    #                             1. Training UNet                                        #
    # =================================================================================== #

    for epoch in range(1, opt.n_epochs + 1):
        train_epoch_loss = 0
        val_epoch_loss = 0

        rmse_epoch = 0
        psnr_epoch = 0
        psnr_epoch_s = 0
        rmse_epoch_s = 0
        psnr_epoch_ns = 0
        rmse_epoch_ns = 0

        mae_epoch = 0
        mae_epoch_s = 0
        mae_epoch_ns = 0

        ssim_epoch = 0
        ssim_epoch_s = 0
        ssim_epoch_ns = 0

        lpips_epoch = 0

        unet = unet.train()
        pbar = tqdm.tqdm(total=train_dataset.__len__(), desc=f"UNet Training Epoch {epoch}/{opt.n_epochs}")
        for i, data in enumerate(train_loader):
            shadow = data['shadow_image']
            shadow_free = data['shadow_free_image']
            mask = data['shadow_mask']

            inp = shadow.type(Tensor).to(device)
            gt = shadow_free.type(Tensor).to(device)
            mask = mask.type(Tensor).to(device)

        
            penumbra_tens = torch.empty_like(mask)
            for i in range(mask.shape[0]):
                penumbra_mask = penumbra(mask[i])
                penumbra_tens[i, :, :, :] = penumbra_mask

            pen_mask = torch.clamp(mask + penumbra_tens, 0, 1)

            penumbra_tens_exp = penumbra_tens.expand(-1, 3, -1, -1) 
            mask_exp = mask.expand(-1, 3, -1, -1)
            pen_mask_exp = pen_mask_exp = pen_mask.expand(-1, 3, -1, -1)

            inp_u = torch.cat((inp, mask, penumbra_tens), dim = 1) #tolta masckera messa mask+pen

            optimizer.zero_grad()
            out_u = unet(inp_u)

            out_f = exposureRGB_Tens(inp, out_u)

            #out_f = torch.where(pen_mask == 0, inp, out_f) #innested only in shadow+penombra region

            out_f = torch.clamp(out_f, 0, 1)
            gt = torch.clamp(gt, 0, 1)

            loss = 1 * criterion_pixelwise(out_f[mask_exp != 0], gt[mask_exp != 0]) + 0.5 * criterion_pixelwise(out_f[penumbra_tens_exp != 0], gt[penumbra_tens_exp != 0]) + 0.4 * criterion_perceptual(out_f, gt) 
            #loss = criterion_pixelwise(out_f[pen_mask_exp != 0], gt[pen_mask_exp != 0])

            loss.backward()  
            optimizer.step()

            train_epoch_loss += loss.detach().item()

            pbar.update(opt.batch_size)
        pbar.close()

        train_loss.append(train_epoch_loss / len(train_loader))
        print(f"[UNet -> Train Loss: {train_epoch_loss / len(train_loader)}]")

        torch.cuda.empty_cache()

        # =================================================================================== #
        #                             2. Validation                                           #
        # =================================================================================== #
        if (epoch) % opt.valid_checkpoint == 0 or epoch == 1:
            with torch.no_grad():
                unet.eval()

                pbar = tqdm.tqdm(total=val_dataset.__len__(), desc=f"Validation Epoch {epoch}/{opt.n_epochs}")
                for idx, data in enumerate(val_loader):
                    shadow = data['shadow_image']
                    shadow_free = data['shadow_free_image']
                    mask = data['shadow_mask']

                    inp = shadow.type(Tensor)
                    gt = shadow_free.type(Tensor)
                    mask = mask.type(Tensor)


                    penumbra_tens = torch.empty_like(mask)
                    for i in range(mask.shape[0]):
                        penumbra_mask = penumbra(mask[i])
                        penumbra_tens[i, :, :, :] = penumbra_mask

                    pen_mask = torch.clamp(mask + penumbra_tens, 0, 1)
                    
                    penumbra_tens_exp = penumbra_tens.expand(-1, 3, -1, -1) 
                    mask_exp = mask.expand(-1, 3, -1, -1)
                    pen_mask_exp = pen_mask.expand(-1, 3, -1, -1)

                    inp_u = torch.cat((inp, mask, penumbra_tens), dim = 1)

                    d_type = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.autocast(device_type=d_type):
                        out_u = unet(inp_u)

                    out_f = exposureRGB_Tens(inp, out_u)

                    #out_f = torch.where(pen_mask == 0, inp, out_f) #innested only in shadow+penombra region

                    out_f = torch.clamp(out_f, 0, 1)
                    gt = torch.clamp(gt, 0, 1)

                    loss = 1 * criterion_pixelwise(out_f[mask_exp != 0], gt[mask_exp != 0]) + 0.5 * criterion_pixelwise(out_f[penumbra_tens_exp != 0], gt[penumbra_tens_exp != 0])+ 0.4 * criterion_perceptual(out_f, gt) 
                    #loss = criterion_pixelwise(out_f[pen_mask_exp != 0], gt[pen_mask_exp != 0])

                    ### METRICS ##########################################################################
                    
                    #PSNR
                    psnr = PSNR_(out_f, gt)
                    psnr_s = PSNR_(out_f*mask_exp, gt*mask_exp)
                    psnr_ns = PSNR_(out_f * (1 - mask_exp), gt * (1 - mask_exp))

                    #RMSE
                    rmse = RMSE_(out_f, gt)
                    rmse_s = RMSE_(out_f*mask_exp, gt*mask_exp)
                    rmse_ns = RMSE_(out_f * (1 - mask_exp), gt * (1 - mask_exp))

                    #LPIPS
                    lpips = criterion_perceptual(out_f, gt)

                    #MAE
                    output_lab = cv2.cvtColor(torch.clamp(out_f[0], 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)), cv2.COLOR_RGB2LAB)
                    gt_lab = cv2.cvtColor(torch.clamp(gt[0], 0, 1).cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2LAB)
                    bm = mask[0].cpu().numpy().transpose(1, 2, 0)

                    mae = np.abs(output_lab - gt_lab).mean()
                    mae_s= np.abs(output_lab * bm - gt_lab * bm).sum() / bm.sum()
                    mae_ns = np.abs(output_lab * (1 - bm) - gt_lab * (1 - bm)).sum() / (1 - bm).sum()

            
                    #SSIM
                    output_g = cv2.cvtColor(torch.clamp(out_f[0], 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)), cv2.COLOR_RGB2GRAY)
                    gt_g = cv2.cvtColor(torch.clamp(gt[0], 0, 1).cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

                    ssim = SSIM_(output_g, gt_g, data_range=1.0, channel_axis=None)
                    ssim_s = SSIM_(output_g * bm.squeeze(), gt_g * bm.squeeze(), data_range=1.0, channel_axis=None)
                    ssim_ns = SSIM_(output_g * (1 - bm.squeeze()), gt_g * (1 - bm.squeeze()), data_range=1.0, channel_axis=None)

                    #
                    
                    val_epoch_loss += loss.detach().item()

                    psnr_epoch += psnr.detach().item()
                    psnr_epoch_s += psnr_s.detach().item()
                    psnr_epoch_ns += psnr_ns.detach().item()

                    rmse_epoch += rmse.detach().item()
                    rmse_epoch_s += rmse_s.detach().item()
                    rmse_epoch_ns += rmse_ns.detach().item()

                    mae_epoch += mae
                    mae_epoch_s += mae_s
                    mae_epoch_ns += mae_ns

                    ssim_epoch += ssim
                    ssim_epoch_s += ssim_s
                    ssim_epoch_ns += ssim_ns

                    lpips_epoch += lpips.detach().item()

                    pbar.update(val_dataset.__len__()/len(val_loader))
                pbar.close()

                if opt.save_images and (epoch == 1 or epoch % 50 == 0):
                    os.makedirs(os.path.join(checkpoint_dir, "images"), exist_ok=True)
                    for count in range(len(shadow)):
                        im_input = (shadow[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        im_pred = (out_f[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        im_gt = (gt[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                        im_input = np.clip(im_input, 0, 255)
                        im_pred = np.clip(im_pred, 0, 255)
                        im_gt = np.clip(im_gt, 0, 255)

                        im_conc = np.concatenate((im_input, im_pred, im_gt), axis=1)
                        im_conc = cv2.cvtColor(im_conc, cv2.COLOR_RGB2BGR)

                        # Save image to disk
                        cv2.imwrite(os.path.join(checkpoint_dir, "images", f"epoch_{epoch}_img_{count}.png"),im_conc)

                val_loss.append(val_epoch_loss / len(val_loader))

                val_rmse.append(rmse_epoch / len(val_loader))
                val_rmse_s.append(rmse_epoch_s / len(val_loader))
                val_rmse_ns.append(rmse_epoch_ns / len(val_loader))

                val_psnr.append(psnr_epoch / len(val_loader))
                val_psnr_s.append(psnr_epoch_s / len(val_loader))
                val_psnr_ns.append(psnr_epoch_ns / len(val_loader))

                val_mae.append(mae_epoch / len(val_loader))
                val_mae_s.append(mae_epoch_s / len(val_loader))
                val_mae_ns.append(mae_epoch_ns / len(val_loader))

                val_ssim.append(ssim_epoch / len(val_loader))
                val_ssim_s.append(ssim_epoch_s / len(val_loader))
                val_ssim_ns.append(ssim_epoch_ns / len(val_loader))

                val_lpips.append(lpips_epoch / len(val_loader))

                if opt.checkpoint_mode != "n":
                    os.makedirs(os.path.join(checkpoint_dir, "weights"), exist_ok=True)
                    if val_epoch_loss < best_loss:
                        best_loss = val_epoch_loss
                        torch.save(unet.state_dict(), os.path.join(checkpoint_dir, "weights", "best_unet_train.pth"))
                    
                    torch.save(unet.state_dict(), os.path.join(checkpoint_dir, "weights", "last_unet_train.pth"))

                    if opt.checkpoint_mode == "all":
                        torch.save(unet.state_dict(), os.path.join(checkpoint_dir, "weights", f"epoch_{epoch}_unet_train.pth"))
  
            #print(f"[Valid Loss: {val_loss[-1]}] [Valid Pix Loss: {val_pix_loss[-1]}] [Valid Perceptual Loss: {val_perceptual_loss[-1]}] [Valid RMSE: {val_rmse[-1]}] [Valid PSNR: {val_psnr[-1]}]")
            print(f"[Valid Loss: {val_loss[-1]}]") #[Valid RMSE: {rmse[-1]}] [Valid PSNR: {psnr[-1]}] [Valid MAE: {mae}] [Valid SSIM: {ssim}] [Valid LPIPS: {lpips[-1]}]]")
        
        #remove in final version
        if epoch == round(0.25 * opt.n_epochs):
            send_telegram_notification(f"25% of training is complete, still {opt.n_epochs - epoch} epochs remaining. :(")
        elif epoch == round(0.5 * opt.n_epochs):
            send_telegram_notification(f"50% of training is complete, still {opt.n_epochs - epoch} epochs remaining. :|")
        elif epoch == round(0.75 * opt.n_epochs):
            send_telegram_notification(f"75% of training is complete, still {opt.n_epochs - epoch} epochs remaining. :)")
        elif (opt.n_epochs - epoch) == 10 and opt.n_epochs != 40:
            send_telegram_notification(f"Only {opt.n_epochs - epoch} epochs remain. :))")



            
        # Save metrics to disk
        metrics_dict = {
            "train_loss_unet": train_loss,
            "val_loss_unet": val_loss,

            "all_PSNR": val_psnr,
            "shadow_PSNR": val_psnr_s,
            "nonshadow_PSNR": val_psnr_ns,
            
            "all_RMSE": val_rmse,
            "shadow_RMSE": val_rmse_s,
            "nonshadow_RMSE": val_rmse_ns,

            "all_MAE":  val_mae,
            "shadow_MAE": val_mae_s,
            "nonshadow_MAE": val_mae_ns,

            "all_SSIM": val_ssim,
            "shadow_SSIM": val_ssim_s,
            "nonshadow_SSIM": val_ssim_ns,

            "LPIPS": val_lpips
        }

        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f)

        torch.cuda.empty_cache()

    print("Training finished")

#Remove in the final version
send_telegram_notification("Hey, Training process is complete!")    