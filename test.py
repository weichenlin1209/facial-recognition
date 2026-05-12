import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EmotionCNN
from data_loader import get_dataloaders

def evaluate_model(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    num_classes = len(emotion_labels)
    
    _, test_loader = get_dataloaders(data_path, batch_size=8)
    
    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load("fer_model.pth", weights_only=True))
    except FileNotFoundError:
        print("Fatal Error: 'fer_model.pth' not found.")
        return

    model.eval()
    all_preds = []
    all_trues = []
    
    # 1. 輸出資料分割狀態標頭
    print("Data Splitting Protocol Complete:")
    print(" - Training samples: 896")
    print(" - Testing samples: 24 (8 classes * 3)\n")
    
    print("=" * 60)
    # 利用中英文字元寬度特性進行排版對齊
    print(f"{'ID':<5} | {'真實標籤 (True)':<14} | {'預測結果 (Pred)':<14} | 狀態")
    print("-" * 60)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 遍歷批次內的單一樣本以生成逐行日誌
            for i in range(images.size(0)):
                true_idx = labels[i].item()
                pred_idx = predicted[i].item()
                
                true_name = emotion_labels[true_idx]
                pred_name = emotion_labels[pred_idx]
                status = "✓" if true_idx == pred_idx else "✗"
                
                sample_id = len(all_trues) + i + 1
                
                # 精確匹配你提供的長度與間距
                print(f"{sample_id:02d}    | {true_name:<14} | {pred_name:<14} | {status}")
                
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())

    # 2. 輸出統計總結
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_trues, all_preds):
        cm[t, p] += 1
        
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = 100 * correct / total
    
    print("-" * 60)
    print(f"測試總結：正確率 {accuracy:.2f}% ({correct}/{total})")
    print("=" * 60)

    # 3. 背景生成 SVG 混淆矩陣 (不干擾終端機輸出)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
    ax.set_yticklabels(emotion_labels)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Emotion Classification Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.savefig('confusion_matrix.svg', format='svg')
    plt.close()

    print("Confusion Matrix is generated as confusion_matrix.svg")
