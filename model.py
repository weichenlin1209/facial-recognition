import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # Conv Block 1: 48x48 -> 24x24
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: 24x24 -> 12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3 (新增): 12x12 -> 6x6，進一步萃取高階語義特徵
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 參數大幅減少: 128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        # 降低 Dropout 比例，避免在小資料集上過度破壞特徵
        self.dropout = nn.Dropout(p=0.3) 
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        # 採用 Leaky ReLU (斜率 0.1) 確保梯度永遠不會歸零
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        
        x = x.view(-1, 128 * 6 * 6)
        
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
