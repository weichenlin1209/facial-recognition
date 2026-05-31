import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from torchvision import transforms
import cv2

class FERDataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.labels = dataframe['emotion'].values
        pixels = [np.fromstring(p, dtype=np.uint8, sep=' ').reshape(48, 48) for p in dataframe['pixels']]
        self.data = pixels
        self.is_train = is_train

        train_transforms_list = [
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
        self.train_transform = transforms.Compose(train_transforms_list)

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_np = self.data[idx]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)

        if self.is_train:
            img_tensor = self.train_transform(img_np)
        else:
            img_tensor = self.test_transform(img_np)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, label_tensor

def get_dataloaders(csv_file, batch_size=32):
    train_df, val_df, test_df = prepare_dataframes(csv_file)

    train_dataset = FERDataset(train_df, is_train=True)
    val_dataset = FERDataset(val_df, is_train=False)
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def prepare_dataframes(csv_file):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_file)

    train_list = []
    test_list = []

    for emotion_class, group in df.groupby('emotion'):
        test_subset = group.iloc[-3:]
        train_subset = group.iloc[:-3]
        test_list.append(test_subset)
        train_list.append(train_subset)

    train_pool = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_pool, test_size=0.2, random_state=42, stratify=train_pool['emotion']
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Data Splitting Protocol Complete:")
    print(f" - Training samples: {len(train_df)}")
    print(f" - Validation samples: {len(val_df)}")
    print(f" - Testing samples: {len(test_df)} (8 classes * 3)")

    return train_df, val_df, test_df
