import argparse
import os
import torch
from dataloader import Dataset
from models.UNet import UNetTranslator_S
from datetime import datetime
from utils.exposure import exposureRGB_Tens
from utils.blurring import penumbra
import numpy as np
import random
import tqdm
import cv2
from torchvision.transforms import v2
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the dataset root")
    parser.add_argument("--unet_size", type=str, default="S", help="size of the UNet model (S, M)")
    parser.add_argument("--model_path", type=str, default="", help="path to UNet weights")
    parser.add_argument("--n_workers", type=int, default=4, help="number of CPU threads for batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use for training, -1 for CPU")
    parser.add_argument("--save_images", type=bool, default=False, help="save images during testing")
    
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
    opt = parse_args()
    print(opt)
    set_seed(42)

    device = torch.device(f"cuda:{opt.gpu}" if opt.gpu >= 0 and opt.gpu < torch.cuda.device_count() else "cpu")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    inf_time = []

    # Create output directory
    checkpoint_dir = os.path.join("output", datetime.now().strftime("%y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.txt"), "w") as f:
        f.write(str(opt))

    # Load UNet model
    unet = UNetTranslator_S(in_channels=5, out_channels=6, deconv=False, local=0).to(device)
    unet.load_state_dict(torch.load(opt.model_path))
    unet.eval()

    Tensor = torch.cuda.FloatTensor if opt.gpu >= 0 else torch.FloatTensor

    # Dataset selection
    if "ISTD" in opt.dataset_path:
        val_dataset = ISTDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True)
    elif "SRD" in opt.dataset_path:
        val_dataset = SRDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=False, masks_precomp=True)
    elif "WRSD" in opt.dataset_path:
        val_dataset = WSRDDataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True, masks_precomp=False)
    else:
        raise ValueError("Data Not Accepted")

    g = torch.Generator()
    g.manual_seed(42)

    test_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=opt.n_workers,
                                              worker_init_fn=seed_worker,
                                              generator=g)
    
    pbar = tqdm.tqdm(total=len(test_loader), desc="Test")
    for idx, data in enumerate(test_loader):
        name_img = data['name'][0]

        inp = data['shadow_image'].type(Tensor).to(device)
        mask = data['shadow_mask'].type(Tensor).to(device)
        gt = data['shadow_free_image'].type(Tensor).to(device)

        start.record()

        penumbra_tens = torch.empty_like(mask)
        for i in range(mask.shape[0]):
            penumbra_mask = penumbra(mask[i])
            penumbra_tens[i, :, :, :] = penumbra_mask

        pen_mask = torch.clamp(mask + penumbra_tens, 0, 1)

        penumbra_tens_exp = penumbra_tens.expand(-1, 3, -1, -1) 
        mask_exp = mask.expand(-1, 3, -1, -1)
        pen_mask_exp = pen_mask_exp = pen_mask.expand(-1, 3, -1, -1)

        inp_u = torch.cat((inp, mask, penumbra_tens), dim = 1)

        out_u = unet(inp_u)

        out_f = exposureRGB_Tens(inp, out_u)

        out_f = torch.where(pen_mask == 0, inp, out_f)

        end.record() #<- END TIMER

        #################
        torch.cuda.synchronize()
        inf_time.append(start.elapsed_time(end)/1000)
        #################

        # Clamping final output
        out_fin = torch.clamp(out_f, 0, 1)

        # Convert to image and save
        im_pred = (out_fin[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im_pred = np.clip(im_pred, 0, 255)
        im_pred = cv2.cvtColor(im_pred, cv2.COLOR_RGB2BGR)

        if opt.save_images:
            os.makedirs(os.path.join(checkpoint_dir, "images"), exist_ok=True)
            cv2.imwrite(os.path.join(checkpoint_dir, "images", str(name_img)), im_pred)
        
        pbar.update(1)
    pbar.close()

    m_time = np.mean(inf_time)
    img_s = 1/m_time
    print(f"\n\n"
          f"- Inference Time ---------------------- \n"
          f"Avg t per Image: {m_time:.4f} s\n"
          f"Img/s: {img_s:.2f}\n"
          f"---------------------------------------")
