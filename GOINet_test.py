import argparse
import os
import torch
from dataloader import Dataset
from models.ResNet import CustomResNet50
from datetime import datetime
from utils.exposure import exposureRGB_Tens
from utils.blurring import dilate_erode_mask
import numpy as np
import random
import tqdm
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the dataset root")

    parser.add_argument("--model_path", type=str, default=False, help="resnet weigth path")
    parser.add_argument("--n_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="gpu to use for training, -1 for cpu")

    parser.add_argument("--save_images", type=bool, default=False, help="save images during test")

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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    inf_time = []

    # Create output directory
    if opt.save_images:
        checkpoint_dir = os.path.join("output", f"Global_{datetime.now().strftime('%y%m%d_%H%M%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.txt"), "w") as f:
            f.write(str(opt))

    # Call trained model
    resnet = CustomResNet50().to(device)
    resnet.load_state_dict(torch.load(opt.model_path))
    resnet.eval()

    Tensor = torch.cuda.FloatTensor if opt.gpu >= 0 else torch.FloatTensor

    val_dataset = Dataset(os.path.join(opt.dataset_path, "test"), aug=False, fix_color=True)
    
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
        name_img = data['name'][0]

        inp = data['shadow_image'].type(Tensor).to(device)
        mask = data['shadow_mask'].type(Tensor).to(device)
        gt = data['shadow_free_image'].type(Tensor).to(device)
        cont_mask = data['contour_mask'].type(Tensor).unsqueeze(1).to(device)

        start.record() #<- START TIMER

        _, eroded_mask = dilate_erode_mask(mask, kernel_size=5)

        exposure_img = torch.zeros_like(inp)
        for c in range(inp.shape[1]):
            j = 2 * c 
            inp_ch = inp[:, c, :, :]
            mu = torch.empty(inp.shape[0], device=device)
            sd = torch.empty(inp.shape[0], device=device)
            for b in range(inp.shape[0]):  
                inp_b = inp[b, c, :, :] 
                mask_b = mask[b, 0, :, :]  
                med = torch.mean(inp_b[mask_b == 1])
                stand = torch.std(inp_b[mask_b == 1])
                mu[b] = med.item()
                sd[b] = stand.item()
            mu = mu.view(-1, 1, 1).to(device)
            sd = sd.view(-1, 1, 1).to(device)

            mu_t = torch.empty(inp.shape[0], device=device)
            sd_t = torch.empty(inp.shape[0], device=device)
            for b in range(inp.shape[0]):  
                inp_b = inp[b, c, :, :] 
                cont_mask_b = cont_mask[b, 0, :, :]  
                med_t = torch.mean(inp_b[cont_mask_b == 1])
                stand_t = torch.std(inp_b[cont_mask_b == 1])
                mu_t[b] = med_t.item()
                sd_t[b] = stand_t.item()
            mu_t = mu_t.view(-1, 1, 1).to(device)
            sd_t = sd_t.view(-1, 1, 1).to(device)

            sd = torch.where(sd == 0, sd + 1e-5, sd)
            exposed_img_ch = mu_t + (inp_ch - mu) * (sd_t / sd)
            exposure_img[:, c, :, :] = exposed_img_ch

        count_img = torch.where(mask == 0., inp, exposure_img)
        
        inp_p = torch.cat((count_img, eroded_mask), dim=1)
        inp_p = torch.clamp(inp_p, 0, 1)

        out_p = resnet(inp_p)

        R_a_mat = out_p[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, count_img.size(2), count_img.size(3))
        R_b_mat = out_p[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, count_img.size(2), count_img.size(3))
        G_a_mat = out_p[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, count_img.size(2), count_img.size(3))
        G_b_mat = out_p[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, count_img.size(2), count_img.size(3))
        B_a_mat = out_p[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, count_img.size(2), count_img.size(3))
        B_b_mat = out_p[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, count_img.size(2), count_img.size(3))

        inp_g = torch.cat((R_a_mat, R_b_mat, G_a_mat, G_b_mat, B_a_mat, B_b_mat), dim=1)
        exp_img = exposureRGB_Tens(count_img, inp_g)
        innested_img = torch.where(mask == 0, inp, exp_img)

        end.record() #<- END TIMER

        #################
        torch.cuda.synchronize()
        inf_time.append(start.elapsed_time(end)/1000)
        #################

        output = torch.clamp(innested_img, 0, 1)

        im_pred = (output[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im_pred = np.clip(im_pred, 0, 255)
        im_pred = cv2.cvtColor(im_pred, cv2.COLOR_RGB2BGR)

        # Save image to disk
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

