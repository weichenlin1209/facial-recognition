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
        
        # Conv Block 3: 12x12 -> 6x6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 新增空間丟棄層 (Spatial Dropout)，隨機丟棄整個特徵圖
        self.spatial_dropout = nn.Dropout2d(p=0.2) 
        
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        # 提高 Dropout 比例至 0.5，強制 FC 層學習穩健特徵
        self.dropout = nn.Dropout(p=0.5) 
        
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x, extract_features=False):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.spatial_dropout(x) # 套用空間雜訊
        
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.spatial_dropout(x) # 套用空間雜訊
        
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        
        x = x.view(-1, 128 * 6 * 6)
        
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        
        if extract_features:
            return x
            
        x = self.fc2(x)
        return x
