import torch
import torch.nn as nn
from torchvision.models import resnet101, resnet50
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights
from torchvision.models.resnet import Bottleneck



class CustomResNet50(torch.nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()

        self.resnet = resnet50()#weights=ResNet50_Weights.IMAGENET1K_V1) 
        
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Define custom layer
        num_features = self.resnet.fc.in_features
        #self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 6)) #nn.Linear(num_features, 6)
        self.resnet.fc = nn.Linear(num_features, 6)


    def forward(self, x):
        x = self.resnet(x)  # Forward pass through the ResNet50 layers 
        return x
    
class CustomResNet101(torch.nn.Module): #mai modificato
    def __init__(self):
        super(CustomResNet101, self).__init__()

        self.resnet = resnet101()#weights=ResNet101_Weights.IMAGENET1K_V1)
        
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Define custom layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 6)

    def forward(self, x):
        x = self.resnet(x)  
        return x


