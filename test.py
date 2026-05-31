import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EmotionCNN
from data_loader import get_dataloaders

def evaluate_model(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    num_classes = len(emotion_labels)

    _, _, test_loader = get_dataloaders(data_path, batch_size=8)

    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load("fer_model.pth", weights_only=True))
    except FileNotFoundError:
        print("Fatal Error: 'fer_model.pth' not found.")
        return

    model.eval()
    all_preds = []
    all_trues = []

    print("=" * 60)
    print(f"{'ID':<5} | {'True Label':<14} | {'Pred Label':<14} | Status")
    print("-" * 60)

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
                status = "\u2713" if true_idx == pred_idx else "\u2717"
                sample_id = len(all_trues) + i + 1
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

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(cax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
    ax.set_yticklabels(emotion_labels)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Emotion Classification Confusion Matrix')

    thresh = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_normalized[i, j]
            ax.text(j, i, f'{val:.2f}',
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.svg', format='svg')
    plt.close()

    print("Confusion Matrix is generated as confusion_matrix.svg")
