import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
import pandas as pd
from PIL import Image
from train import train_model
from model_pretrained import model_pretrained
from model_unpretrained import model_unpretrained
import torch.optim as optim


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): CSV 文件的路径，包含图像路径和标签。
            root_dir (string): 图像文件的根目录。
            transform (callable, optional): 需要应用于样本的可选转换。
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir.replace("\\", "/")
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_name'])
        img_name = img_name.replace("\\", "/")  # 确保文件路径使用正斜杠
        image = Image.open(img_name).convert('RGB')

        label = self.data_frame.iloc[idx]['class_id']-1
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device)

        if self.transform:
            image = self.transform(image)

        return image, label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 假设你的 CSV 文件和图像都在上一级目录中
train_csv_file = 'train.csv'
test_csv_file = 'test.csv'
root_dir = 'CUB_200_2011/images'  # 假设所有图像都存储在这个目录下

train_dataset = CustomDataset(csv_file=train_csv_file, root_dir=root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(csv_file=test_csv_file, root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_writer = SummaryWriter('logs/train_b')
test_writer = SummaryWriter('logs/test_b')

if __name__=='__main__':
    num_epochs=30
    criterion=nn.CrossEntropyLoss()
    model=model_pretrained
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3},  # 只为全连接层设置较高的学习率
    ], lr=1e-4)
    train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs)

