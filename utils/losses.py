import torch
import torch.nn as nn

import ipdb

class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def adjust_dimensions(self, x):
        # squeeze useless dimensions (if c=1)
        x = x.squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def __call__(self, pred, gt):
        pred = self.adjust_dimensions(pred)
        gt = self.adjust_dimensions(gt)

        # compute intersection for all batch elements (b, h, w)
        intersection = torch.sum(pred * gt, dim=(1, 2))
        # compute union for all batch elements (b, h, w)
        union = torch.sum(pred + gt, dim=(1, 2)) - intersection
        # compute iou for all batch elements (b)
        iou = intersection / union
        return torch.mean(iou)
