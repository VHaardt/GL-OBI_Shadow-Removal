import argparse
import os
import torch
from dataloader import ISTDDataset, SRDDataset, WSRDDataset
from models.UNet import UNetTranslator_S
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
from torchvision.transforms import v2

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the dataset root")

    parser.add_argument("--unet_size", type=str, default="S", help="size of the model (S, M)")
    parser.add_argument("--model_path", type=str, default=False, help="model weigth path")

    parser.add_argument("--n_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="gpu to use for training, -1 for cpu")

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
    set_seed(42)

    device = torch.device(f"cuda:{opt.gpu}" if opt.gpu >= 0 and opt.gpu >= torch.cuda.device_count() - 1 else "cpu")

    # Create output directory
    checkpoint_dir = os.path.join("output", datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.txt"), "w") as f:
        f.write(str(opt))
    
    unet = UNetTranslator_S(in_channels=4, out_channels=3, deconv=False, local=0).to(device) #residual settato False
    unet.load_state_dict(torch.load(opt.model_path))
    unet.eval()

    Tensor = torch.cuda.FloatTensor if opt.gpu >= 0 else torch.FloatTensor

    if "ISTD" in opt.dataset_path:
        val_dataset = ISTDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True)
    elif "SRD" in opt.dataset_path:
        val_dataset = SRDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=False, masks_precomp=True)
    elif "WRSD" in opt.dataset_path:
        val_dataset = WSRDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True, masks_precomp=False)

    g = torch.Generator()
    g.manual_seed(42)

    test_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=opt.n_workers,
                                                    worker_init_fn=seed_worker,
                                                    generator=g)
    
    pbar = tqdm.tqdm(total=test_loader.__len__(), desc=f"Test")
    for idx, data in enumerate(test_loader):
        shadow = data['shadow_image']
        shadow_free = data['shadow_free_image']
        mask = data['shadow_mask']
        name_img = data['name'][0]
        exp_img = data['exposed_image']

        inp = shadow.type(Tensor)
        gt = shadow_free.type(Tensor)
        mask = mask.type(Tensor)
        exp_img = exp_img.type(Tensor)

        innested_img = torch.where(mask == 0., inp, exp_img)

        penumbra_tens = torch.empty_like(mask)
        for i in range(mask.shape[0]):
            penumbra_mask = penumbra(mask[i])
            penumbra_tens[i, :, :, :] = penumbra_mask

        pen_mask = torch.clamp(mask + penumbra_tens, 0, 1)

        penumbra_tens_exp = penumbra_tens.expand(-1, 3, -1, -1) 
        mask_exp = mask.expand(-1, 3, -1, -1)
        pen_mask_exp = pen_mask_exp = pen_mask.expand(-1, 3, -1, -1)

        inp_u = torch.cat((innested_img, penumbra_tens), dim = 1)

        out_u = unet(inp_u)

        out_f = exposure_3ch(innested_img, out_u)

        out_f = torch.clamp(out_f, 0, 1)
        gt = torch.clamp(gt, 0, 1)

        im_pred = (out_f[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im_pred = np.clip(im_pred, 0, 255)
        im_pred = cv2.cvtColor(im_pred, cv2.COLOR_RGB2BGR)

        # Save image to disk
        os.makedirs(os.path.join(checkpoint_dir, "images"), exist_ok=True)
        cv2.imwrite(os.path.join(checkpoint_dir, "images", str(name_img)), im_pred)

        pbar.update(1)
    pbar.close()

