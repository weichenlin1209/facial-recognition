import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import EmotionCNN
from data_loader import get_dataloaders

def evaluate_current_model(model, test_loader, device):
    """內部驗證常式：於訓練週期中即時計算驗證集正確率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0

def train_model(data_path, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Allocating computations to: {device}")
    
    model = EmotionCNN().to(device)
    train_loader, test_loader = get_dataloaders(data_path)
    
    criterion = nn.CrossEntropyLoss()
    # 引入 weight_decay 進行 L2 正則化 (L2 Regularization) 以對抗過擬合
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    loss_history = []
    best_accuracy = 0.0
    
    print(f"Initiating Training Sequence: {epochs} Epochs")
    for epoch in range(epochs):
        # 確保模型處於訓練模式
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
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # 即時驗證與最佳狀態捕捉
        current_accuracy = evaluate_current_model(model, test_loader, device)
        
        # 為了保持終端機整潔，僅在遇到最佳解或每 5 個 Epoch 時輸出狀態
        if current_accuracy > best_accuracy or (epoch + 1) % 5 == 0 or epoch == 0:
            log_str = f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Val Acc: {current_accuracy:.2f}%"
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                # 僅在正確率突破歷史極值時，將張量狀態寫入硬碟
                torch.save(model.state_dict(), "fer_model.pth")
                log_str += " -> [Checkpoint Updated]"
                
            print(log_str)
    
    # --- 繪圖與輸出 SVG 子程序 ---
    print("\nOptimization sequence terminated.")
    print(f"Global Maximum Validation Accuracy: {best_accuracy:.2f}%")
    print("Generating Learning Curve SVG...")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker='', linestyle='-', color='b')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('learning_curve.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Data exported successfully: learning_curve.svg")
