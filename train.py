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
    train_loader, val_loader, _ = get_dataloaders(data_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    loss_history = []
    best_accuracy = 0.0
    epochs_no_improve = 0
    early_stop_patience = 20

    print(f"Initiating Training Sequence: {epochs} Epochs")
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

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        current_accuracy = evaluate_current_model(model, val_loader, device)

        if current_accuracy > best_accuracy or (epoch + 1) % 5 == 0 or epoch == 0:
            log_str = f"Epoch {epoch+1:03d}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Val Acc: {current_accuracy:.2f}%"

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                epochs_no_improve = 0
                torch.save(model.state_dict(), "fer_model.pth")
                log_str += " -> [Checkpoint Updated]"
            else:
                epochs_no_improve += 1

            print(log_str)

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs).")
            break
    
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
