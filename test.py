import torch
from model import EmotionCNN
from data_loader import get_dataloaders

def evaluate_model(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    
    _, test_loader = get_dataloaders(data_path, batch_size=8)
    
    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load("fer_model.pth", weights_only=True))
    except FileNotFoundError:
        print("邏輯錯誤：找不到權重檔，請先執行訓練 (train) 模式。")
        return

    model.eval()
    
    correct = 0
    total = 0
    
    print(f"\n{'='*60}")
    print(f"{'ID':<5} | {'真實標籤 (True)':<12} | {'預測結果 (Pred)':<12} | {'狀態'}")
    print(f"{'-'*60}")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                true_idx = labels[i].item()
                pred_idx = predicted[i].item()
                
                true_name = emotion_labels[true_idx]
                pred_name = emotion_labels[pred_idx]
                
                status = "✓" if true_idx == pred_idx else "✗"
                
                sample_id = total + i + 1
                print(f"{sample_id:02d}    | {true_name:<12} | {pred_name:<12} | {status}")
                
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"{'-'*60}")
    print(f"測試總結：正確率 {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*60}\n")
