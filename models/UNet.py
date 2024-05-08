import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np

import ipdb

class UNetCompress(nn.Module):
    def __init__(self, in_size, out_size, normalize=True,  kernel_size=4, dropout=0.0):
        super(UNetCompress, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetDecompress(nn.Module):
    def __init__(self, in_size, out_size, deconv = False ,dropout=0.0):
        super(UNetDecompress, self).__init__()
        if deconv:
            layers = [
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True),
        ]
        else:
            layers = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        f = self.model(x)
        o = torch.cat((f, skip_input), 1)
        return o
 

class UNetTranslator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, deconv=False, local=0, residual=True):
        self.local_transform = local
        super(UNetTranslator, self).__init__()
        self.res = residual

        self.down1 = UNetCompress(in_channels, 64, kernel_size=4, normalize=False)
        self.down2 = UNetCompress(64, 128)
        self.down3 = UNetCompress(128, 256)
        self.down4 = UNetCompress(256, 512, dropout=0.25)
        self.down5 = UNetCompress(512, 512, dropout=0.25)
        self.down6 = UNetCompress(512, 512, dropout=0.5)
        self.down7 = UNetCompress(512, 512, dropout=0.5)
        self.down8 = UNetCompress(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetDecompress(512, 512, deconv=deconv, dropout=0.5)
        self.up2 = UNetDecompress(1024, 512, deconv=deconv, dropout=0.5)
        self.up3 = UNetDecompress(1024, 512, deconv=deconv, dropout=0.25)
        self.up4 = UNetDecompress(1024, 512, deconv=deconv, dropout=0.25)
        self.up5 = UNetDecompress(1024, 256, deconv=deconv)
        self.up6 = UNetDecompress(512, 128, deconv=deconv)
        self.up7 = UNetDecompress(256, 64, deconv=deconv)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1)
        )
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        
        _, _, h, w = x.shape
        target_h = math.floor(math.ceil(h / 256) * 256) # x should be divisible by 256
        target_w = math.floor(math.ceil(w / 256) * 256) # w should be divisible by 256
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear')

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        result = self.final(u7)

        if self.res:
            result = torch.clamp(result + x[:,:3,:,:], 0, 1)
        else:
            result = torch.linear(result)

        result = F.interpolate(result, size=(h, w), mode='bilinear')
        return result


class UNetTranslator_S(nn.Module):
    """
        Small version of the UNetTranslator (4 layers instead of 8)
    """
    def __init__(self, in_channels=3, out_channels=3, deconv=False, local=0, residual=True):
        self.local_transform = local
        super(UNetTranslator_S, self).__init__()
        self.res = residual
        
        self.down1 = UNetCompress(in_channels, 64, kernel_size=4, normalize=False)
        self.down2 = UNetCompress(64, 128)
        self.down3 = UNetCompress(128, 256)
        self.down4 = UNetCompress(256, 512, dropout=0.25)

        self.up1 = UNetDecompress(512, 256, deconv=deconv, dropout=0.25)
        self.up2 = UNetDecompress(512, 128, deconv=deconv)
        self.up3 = UNetDecompress(256, 64, deconv=deconv)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1)
        )


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        
        _, _, h, w = x.shape
        target_h = math.floor(math.ceil(h / 16) * 16)
        target_w = math.floor(math.ceil(w / 16) * 16)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear')

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        result = self.final(u3)

        if self.res:
            result = torch.clamp(result + x[:,:3,:,:], 0, 1)
        else:
            result = torch.linear(result)

        result = F.interpolate(result, size=(h, w), mode='bilinear')

        return result
