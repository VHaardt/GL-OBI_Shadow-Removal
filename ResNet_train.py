'''
This section want to test the theory limit of pre training part and the epoch necessary to reach it.
So it will be train and tested on the same dataset.
'''

import argparse
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from dataloader import ISTDDataset
from models.ResNet import CustomResNet101, CustomResNet50
from torchvision.models import resnet101
from datetime import datetime
from utils.metrics import PSNR, RMSE
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

    parser.add_argument("--resnet_size", type=str, default="S", help="size of the model (S, M)")
    parser.add_argument("--resnet_freeze", type=bool, default=False, help="freeze layers of resnet")

    parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="gpu to use for training, -1 for cpu")

    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")#0.0002
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--decay_epoch", type=int, default=400, help="epoch from which to start lr decay")
    parser.add_argument("--decay_steps", type=int, default=4, help="number of step decays")

    parser.add_argument("--pixel_weight", type=float, default=1, help="weight of the pixelwise loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.5, help="weight of the perceptual loss")

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
    response = requests.post(url, json=payload)
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

    # Define ResNet
    if opt.resnet_size == "S":
        resnet = CustomResNet50(freeze = opt.resnet_freeze).to(device)
    else:
        resnet = CustomResNet101(freeze = opt.resnet_freeze).to(device)
        
    
    n_params1 = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"Model 1 has {n_params1} trainable parameters")
    with open(os.path.join(checkpoint_dir, "architecture.txt"), "w") as f:
        f.write("Model has " + str(n_params1) + " trainable parameters\n")
        f.write(str(resnet))
    
    # Define loss
    #criterion_pixelwise = torch.nn.MSELoss().to(device)
    criterion_pixelwise = torch.nn.L1Loss().to(device)
    criterion_perceptual = LPIPS(net_type="vgg" , normalize= True).to(device)
    criterion_segm = torch.nn.BCEWithLogitsLoss().to(device)


    # Define metrics
    PSNR_ = PSNR()
    RMSE_ = RMSE()

    # Define optimizer for Resnet
    optimizer_p = torch.optim.Adam(resnet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optimizer_p, gamma=0.01) 
    #decay_step_p = (opt.n_epochs - opt.decay_epoch) // opt.decay_steps
    #milestones_p = [opt.decay_epoch + i * opt.decay_steps for i in range((opt.n_epochs - opt.decay_epoch) // opt.decay_steps)]
    #scheduler_p = MultiStepLR(optimizer_p, milestones=milestones_p, gamma=0.5)

    Tensor = torch.cuda.FloatTensor if opt.gpu >= 0 else torch.FloatTensor

    # Train and test on the same data
    if "ISTD" in opt.dataset_path:
        train_dataset = ISTDDataset(os.path.join(opt.dataset_path, "train"), size=(opt.img_height, opt.img_width), aug=True, fix_color=True)
        val_dataset = ISTDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True)

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
    
    pretrain_loss = []
    pretrain_val_loss = []

    pretrain_val_rmse = []
    pretrain_val_psnr = []
    pretrain_val_psnr_shadow = []
    pretrain_val_rmse_shadow = []

    best_loss = 1e3 # arbitrary large number


    # =================================================================================== #
    #                             1. Pre-training of ResNet                               #
    # =================================================================================== #

    for epoch in range(1, opt.n_epochs + 1):
        pretrain_epoch_loss = 0

        pretrain_valid_epoch_loss = 0

        pretrain_rmse_epoch = 0
        pretrain_psnr_epoch = 0
        pretrain_psnr_epoch_shadow = 0
        pretrain_rmse_epoch_shadow = 0

        resnet.train()
        pbar = tqdm.tqdm(total=train_dataset.__len__(), desc=f"ResNet pre-training Epoch {epoch}/{opt.n_epochs}")
        for i, data in enumerate(train_loader):
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

            optimizer_p.zero_grad()
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

            l_lpips = []
            for j in range(blurred_innested_img.shape[0]):
                a = blurred_innested_img[j]
                b = gt[j]
                crop_img = a[:, int(crop_coordinate[j][0]):int(crop_coordinate[j][1]), int(crop_coordinate[j][2]):int(crop_coordinate[j][3])]
                crop_gt = b[:, int(crop_coordinate[j][0]):int(crop_coordinate[j][1]), int(crop_coordinate[j][2]):int(crop_coordinate[j][3])]
                crop_img = crop_img.expand(1, -1, -1, -1)
                crop_gt = crop_gt.expand(1, -1, -1, -1)
                lpips = criterion_perceptual(crop_img, crop_gt)
                l_lpips.append(lpips)
            perceptual_loss = torch.stack(l_lpips).mean()

            loss = opt.pixel_weight *  criterion_pixelwise(transformed_images[mask_exp != 0], gt[mask_exp != 0]) + opt.perceptual_weight * perceptual_loss #cambiato con blurred/innested

            loss.backward()  
            optimizer_p.step()

            pretrain_epoch_loss += loss.detach().item()

            pbar.update(opt.batch_size)
        pbar.close()

        pretrain_loss.append(pretrain_epoch_loss / len(train_loader))
        print(f"[ResNet -> Train Loss: {pretrain_epoch_loss / len(train_loader)}]")

        torch.cuda.empty_cache() ###################

        # =================================================================================== #
        #                             2. Validation                                           #
        # =================================================================================== #
        if (epoch-1) % opt.valid_checkpoint == 0:
            with torch.no_grad():
                resnet = resnet.eval()

                pbar = tqdm.tqdm(total=val_dataset.__len__(), desc=f"Validation Epoch {epoch}/{opt.n_epochs}")
                for idx, data in enumerate(val_loader):
                    shadow = data['shadow_image']
                    shadow_free = data['shadow_free_image']
                    mask = data['shadow_mask']
                    crop_coordinate = data['crop_coordinate']

                    inp = shadow.type(Tensor)
                    gt = shadow_free.type(Tensor)
                    mask = mask.type(Tensor)
                    crop_coordinate = crop_coordinate.type(Tensor)
                    
                    inp = torch.clamp(inp, 0, 1)
                    gt = torch.clamp(gt, 0, 1)

                    d_type = "cuda" if torch.cuda.is_available() else "cpu"
                    with torch.autocast(device_type=d_type):
                        out_p = resnet(inp)
                    transformed_images = exposureRGB(inp, out_p)
                    mask_exp = mask.expand(-1, 3, -1, -1)

                    #function to create image where shadow part is form resnet, non shadow part is form input.
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

                    l_lpips = []
                    for j in range(blurred_innested_img.shape[0]):
                        a = blurred_innested_img[j]
                        b = gt[j]
                        crop_img = a[:, int(crop_coordinate[j][0]):int(crop_coordinate[j][1]), int(crop_coordinate[j][2]):int(crop_coordinate[j][3])]
                        crop_gt = b[:, int(crop_coordinate[j][0]):int(crop_coordinate[j][1]), int(crop_coordinate[j][2]):int(crop_coordinate[j][3])]
                        crop_img = crop_img.expand(1, -1, -1, -1)
                        crop_gt = crop_gt.expand(1, -1, -1, -1)
                        lpips = criterion_perceptual(crop_img, crop_gt)
                        l_lpips.append(lpips)
                    perceptual_loss = torch.stack(l_lpips).mean()

                    loss = opt.pixel_weight * criterion_pixelwise(transformed_images[mask_exp != 0], gt[mask_exp != 0]) + opt.perceptual_weight * perceptual_loss

                    psnr = PSNR_(blurred_innested_img, gt)
                    rmse = RMSE_(blurred_innested_img, gt)

                    psnr_shadow = PSNR_(blurred_innested_img*mask_exp, gt*mask_exp)
                    rmse_shadow = RMSE_(blurred_innested_img*mask_exp, gt*mask_exp)
                    
                    pretrain_valid_epoch_loss += loss.detach().item()

                    pretrain_psnr_epoch += psnr.detach().item()
                    pretrain_psnr_epoch_shadow += psnr_shadow.detach().item()
                    pretrain_rmse_epoch += rmse.detach().item()
                    pretrain_rmse_epoch_shadow += rmse_shadow.detach().item()

                    pbar.update(val_dataset.__len__()/len(val_loader))
                pbar.close()

                if opt.save_images: #and (epoch == 1 or epoch % 10 == 0):
                    os.makedirs(os.path.join(checkpoint_dir, "images"), exist_ok=True)
                    for count in range(len(shadow)):
                        im_input = (shadow[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        im_pred = (transformed_images[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        im_innest = (innested_img[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        im_blur = (blurred_innested_img[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        im_gt = (gt[count].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8) #gt

                        im_input = np.clip(im_input, 0, 255) #inserimento per rimuovere astrazioni... CONTROLLARE
                        im_pred = np.clip(im_pred, 0, 255)
                        im_innest = np.clip(im_innest, 0, 255)
                        im_blur = np.clip(im_blur, 0, 255)
                        im_gt = np.clip(im_gt, 0, 255)

                        im_conc = np.concatenate((im_input, im_pred, im_innest, im_blur, im_gt), axis=1)
                        im_conc = cv2.cvtColor(im_conc, cv2.COLOR_RGB2BGR)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(im_conc, str(np.round(out_p[0].tolist(),2)), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        ###

                        # Save image to disk
                        cv2.imwrite(os.path.join(checkpoint_dir, "images", f"epoch_{epoch}_img_{count}.png"),im_conc)

                pretrain_val_loss.append(pretrain_valid_epoch_loss / len(val_loader))
                pretrain_val_rmse.append(pretrain_rmse_epoch / len(val_loader))
                pretrain_val_rmse_shadow.append(pretrain_rmse_epoch_shadow / len(val_loader))
                pretrain_val_psnr.append(pretrain_psnr_epoch / len(val_loader))
                pretrain_val_psnr_shadow.append(pretrain_psnr_epoch_shadow / len(val_loader))

                if opt.checkpoint_mode != "n":
                    os.makedirs(os.path.join(checkpoint_dir, "weights"), exist_ok=True)
                    if pretrain_valid_epoch_loss < best_loss:
                        best_loss = pretrain_valid_epoch_loss
                        torch.save(resnet.state_dict(), os.path.join(checkpoint_dir, "weights", "best_resnet_pretrain.pth"))
                    
                    torch.save(resnet.state_dict(), os.path.join(checkpoint_dir, "weights", "last_resnet_pretrain.pth"))

                    if opt.checkpoint_mode == "all":
                        torch.save(resnet.state_dict(), os.path.join(checkpoint_dir, "weights", f"epoch_{epoch}_resnet_pretrain.pth"))

                

                
            #print(f"[Valid Loss: {val_loss[-1]}] [Valid Pix Loss: {val_pix_loss[-1]}] [Valid Perceptual Loss: {val_perceptual_loss[-1]}] [Valid RMSE: {val_rmse[-1]}] [Valid PSNR: {val_psnr[-1]}]")
            print(f"[Valid Loss: {pretrain_val_loss[-1]}] [Valid RMSE: {pretrain_val_rmse[-1]}] [Valid PSNR: {pretrain_val_psnr[-1]}] [Valid PSNR Shadow: {pretrain_val_psnr_shadow[-1]} [Valid RMSE Shadow: {pretrain_val_rmse_shadow[-1]}]]")

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
            "train_loss_resnet": pretrain_loss,
            "val_loss_resnet": pretrain_val_loss,
            "all_RMSE": pretrain_val_rmse,
            "shadow_RMSE": pretrain_val_rmse_shadow,
            "all_PSNR": pretrain_val_psnr,
            "shadow_PSNR": pretrain_val_psnr_shadow
        }

        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f)

        torch.cuda.empty_cache()    #################

    print("Training finished")

#Remove in the final version
send_telegram_notification("Hey, Training process is complete!")    