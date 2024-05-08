import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet101(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
new_model.add_module('fc', nn.Linear(2048, 6))
