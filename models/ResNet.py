import torch
import torch.nn as nn
from torchvision.models import resnet101, resnet50
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights
from torchvision.models.resnet import Bottleneck

class CustomActivation(nn.Module):
    def forward(self, x):
        x = x.clone()  # Make a copy of x to avoid inplace operation issues
        x[(x == 1) | (x == 3) | (x == 5)] = torch.clamp(x[(x == 1) | (x == 3) | (x == 5)], 1, 10)
        x[(x == 2) | (x == 4) | (x == 6)] = torch.clamp(x[(x == 2) | (x == 4) | (x == 6)], -0.1, 0.5)
        return x


class CustomResNet50(torch.nn.Module):
    def __init__(self, freeze = False):
        super(CustomResNet50, self).__init__()
        self.freeze = freeze

        self.resnet = resnet50(weights=None)#weights=ResNet50_Weights.IMAGENET1K_V1)

        if self.freeze: # Freeze all layers except the last one
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Define custom layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 6)) #nn.Linear(num_features, 6)
        self.custom_activation = CustomActivation()

    def forward(self, x):
        x = self.resnet(x)  # Forward pass through the ResNet50 layers 
        #x = self.custom_activation(x)
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
        self.resnet.fc = nn.Linear(num_features, 6)

    def forward(self, x):
        x = self.resnet(x)  
        return x


