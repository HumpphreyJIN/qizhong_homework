import torch
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 全部权重从初始化训练
model_unpretrained = models.resnet18(pretrained=False).to(device)  # 使用随机初始化的权重

# 确保所有参数都是可训练的
for param in model_unpretrained.parameters():
    param.requires_grad = True

# 修改最后的全连接层
model_unpretrained.fc = nn.Linear(model_unpretrained.fc.in_features, 200)
