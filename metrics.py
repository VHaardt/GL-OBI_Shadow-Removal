import torch
import ipdb

class PSNR():
    def __init__(self):
        pass

    def adjust_dimensions(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    def __call__(self, ref, res):
        ref = self.adjust_dimensions(ref)
        res = self.adjust_dimensions(res)
        
        mse = torch.mean((ref - res) ** 2, dim=(1, 2, 3)) # mse for each element in the batch (b, c, h, w)
        psnr = 10 * torch.log10(1 / mse) # psnr for each mse in the batch

        return torch.mean(psnr)


class SSIM():
    def __init__(self):
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def adjust_dimensions(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    def __call__(self, ref, res):
        ref = self.adjust_dimensions(ref)
        res = self.adjust_dimensions(res)
        
        mu_x = torch.mean(ref, dim=(1, 2, 3))
        mu_y = torch.mean(res, dim=(1, 2, 3))
        sigma_x = torch.var(ref, dim=(1, 2, 3))
        sigma_y = torch.var(res, dim=(1, 2, 3))
        sigma_xy = torch.mean((ref - mu_x) * (res - mu_y), dim=(1, 2, 3))

        ssim = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2) / ((mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2))
        return torch.mean(ssim)


class RMSE():
    def __init__(self):
        pass

    def adjust_dimensions(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    def __call__(self, ref, res):
        ref = self.adjust_dimensions(ref)
        res = self.adjust_dimensions(res)
        
        mse = torch.mean((ref - res) ** 2, dim=(1, 2, 3)) # mse for each element in the batch (b, c, h, w)
        rmse = torch.sqrt(mse) # rmse for each mse in the batch

        return torch.mean(rmse)
