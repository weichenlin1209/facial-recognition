import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # 新增匯入
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

class FERDataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.labels = dataframe['emotion'].values
        pixels = [np.fromstring(p, dtype=np.uint8, sep=' ').reshape(48, 48) for p in dataframe['pixels']]
        self.data = pixels
        self.is_train = is_train
        
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 1. 取得原始 NumPy 陣列
        img_np = self.data[idx]
        
        # 2. 注入 CLAHE 光學濾波器 (增強對比度)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_np = clahe.apply(img_np)
        
        # 3. 根據狀態套用轉換管線
        if self.is_train:
            img_tensor = self.train_transform(img_np)
        else:
            img_tensor = self.test_transform(img_np)
            
        # 4. 提取對應的標籤並轉換為張量 (Tensor)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 5. 必須明確回傳包含特徵與標籤的 Tuple
        return img_tensor, label_tensor

def get_dataloaders(csv_file, batch_size=32):
    train_df, test_df = prepare_dataframes(csv_file)
    
    train_dataset = FERDataset(train_df, is_train=True)
    test_dataset = FERDataset(test_df, is_train=False)
    
    train_labels = train_df['emotion'].values
    
    class_counts = np.bincount(train_labels)
    
    class_weights = 1.0 / class_counts
    
    sample_weights = np.array([class_weights[t] for t in train_labels])
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    target_samples_per_epoch = 8 * 300
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=target_samples_per_epoch,
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def prepare_dataframes(csv_file):
    """
    自定義分割邏輯：每種情緒類別的最後 3 張作為測試集，其餘為訓練集。
    """
    df = pd.read_csv(csv_file)
    
    train_list = []
    test_list = []
    
    for emotion_class, group in df.groupby('emotion'):
        test_subset = group.iloc[-3:]
        train_subset = group.iloc[:-3]
        
        test_list.append(test_subset)
        train_list.append(train_subset)
        
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    print(f"Data Splitting Protocol Complete:")
    print(f" - Training samples: {len(train_df)}")
    print(f" - Testing samples: {len(test_df)} (8 classes * 3)")
    
    return train_df, test_df

