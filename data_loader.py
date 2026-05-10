import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms # 新增匯入
from PIL import Image              # 新增匯入

class FERDataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.labels = dataframe['emotion'].values
        pixels = [np.fromstring(p, dtype=np.uint8, sep=' ').reshape(48, 48) for p in dataframe['pixels']]
        self.data = pixels
        self.is_train = is_train
        
        # 建立訓練期的增強管線
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),       # 隨機旋轉正負 10 度
            # transforms.RandomHorizontalFlip(),   # 50% 機率水平翻轉
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 隨機平移 10%
            transforms.ToTensor()                # 轉換為 Tensor 並自動除以 255 歸一化
        ])
        
        # 測試期保持唯讀，僅作張量轉換
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_np = self.data[idx]
        
        # 根據是否為訓練集，應用不同的轉換邏輯
        if self.is_train:
            img_tensor = self.train_transform(img_np)
        else:
            img_tensor = self.test_transform(img_np)
            
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor

# 在 get_dataloaders 中實體化時傳入 is_train 參數
def get_dataloaders(csv_file, batch_size=32):
    train_df, test_df = prepare_dataframes(csv_file)
    train_dataset = FERDataset(train_df, is_train=True)
    test_dataset = FERDataset(test_df, is_train=False)
    
    train_dataset = FERDataset(train_df)
    test_dataset = FERDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    
def prepare_dataframes(csv_file):
    """
    自定義分割邏輯：每種情緒類別的最後 3 張作為測試集，其餘為訓練集。
    """
    df = pd.read_csv(csv_file)
    
    train_list = []
    test_list = []
    
    # 根據 'emotion' 欄位進行群組化
    for emotion_class, group in df.groupby('emotion'):
        # Python 切片語法：[-3:] 代表取最後 3 筆，[:-3] 代表取除了最後 3 筆以外的所有資料
        test_subset = group.iloc[-3:]
        train_subset = group.iloc[:-3]
        
        test_list.append(test_subset)
        train_list.append(train_subset)
        
    # 將切割後的子集重新合併為單一的 DataFrame
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    # 輸出狀態驗證
    print(f"Data Splitting Protocol Complete:")
    print(f" - Training samples: {len(train_df)}")
    print(f" - Testing samples: {len(test_df)} (8 classes * 3)")
    
    return train_df, test_df

