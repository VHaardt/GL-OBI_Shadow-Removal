import torch
import torch.nn as nn
from torchvision.models import resnet101, resnet50
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights
from torchvision.models.resnet import Bottleneck



class CustomResNet50(torch.nn.Module):
    def __init__(self, freeze = False):
        super(CustomResNet50, self).__init__()
        self.freeze = freeze

        self.resnet = resnet50(pretrained=True)

        if self.freeze: # Freeze all layers except the last one
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Define custom layer
        num_features = self.resnet.fc.in_features
        self.custom_fc = nn.Linear(num_features, 6)

    def forward(self, x):
        x = self.resnet(x)  # Forward pass through the ResNet50 layers
            
        channel1 = x[:, 0].clamp(1, 10)
        channel2 = x[:, 1].clamp(-0.5, 0.5)
        channel3 = x[:, 2].clamp(1, 10)
        channel4 = x[:, 3].clamp(-0.5, 0.5)
        channel5 = x[:, 4].clamp(1, 10)
        channel6 = x[:, 5].clamp(-0.5, 0.5)
        
        x = torch.cat((channel1.unsqueeze(1), channel2.unsqueeze(1), channel3.unsqueeze(1),
                    channel4.unsqueeze(1), channel5.unsqueeze(1), channel6.unsqueeze(1)), dim=1) 
        return x
class CustomResNet101(torch.nn.Module):
    def __init__(self, freeze = False):
        super(CustomResNet101, self).__init__()
        self.freeze = freeze

        self.resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        

        if self.freeze: # Freeze all layers except the last one
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Define custom layer
        num_features = self.resnet.fc.in_features
        self.custom_fc = nn.Linear(num_features, 6)

    def forward(self, x):
        x = self.resnet(x) 

        channel1 = x[:, 0].clamp(1, 10)
        channel2 = x[:, 1].clamp(-0.5, 0.5)
        channel3 = x[:, 2].clamp(1, 10)
        channel4 = x[:, 3].clamp(-0.5, 0.5)
        channel5 = x[:, 4].clamp(1, 10)
        channel6 = x[:, 5].clamp(-0.5, 0.5)
        
        x = torch.cat((channel1.unsqueeze(1), channel2.unsqueeze(1), channel3.unsqueeze(1),
                    channel4.unsqueeze(1), channel5.unsqueeze(1), channel6.unsqueeze(1)), dim=1) 
        return x


