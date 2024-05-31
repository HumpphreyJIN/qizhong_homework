import torch
from torchvision import models
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pretrained = models.resnet18(pretrained=True).to(device)
for param in model_pretrained.parameters():
    param.requires_grad = False

model_pretrained.fc = nn.Linear(model_pretrained.fc.in_features, 200)
for param in model_pretrained.fc.parameters():
    param.requires_grad = True
