import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # 新增匯入
from model import EmotionCNN
from data_loader import get_dataloaders

def train_model(data_path, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Allocating computations to: {device}")
    
    model = EmotionCNN().to(device)
    train_loader, val_loader = get_dataloaders(data_path)
    
    # --- [修正 1] 動態計算類別權重 (Class Weights) ---
    print("Calculating Class Weights to mitigate imbalance...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    # Inverse Frequency Weighting Formula
    class_weights = total_samples / (num_classes * class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 將權重張量注入損失函數
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    # -------------------------------------------------
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # --- [修正 2] 學習率排程器 (Learning Rate Scheduler) ---
    # 每 15 個 Epoch，將學習率乘以 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 推進排程器
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
    
    torch.save(model.state_dict(), "fer_model.pth")
    print("Model state dumped to fer_model.pth")
