import torch
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from model import EmotionCNN
from data_loader import get_dataloaders

EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (48, 48))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return img_tensor

def predict_cnn(image_path, cnn_checkpoint="fer_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(cnn_checkpoint, weights_only=True))
    model.eval()

    img_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return EMOTION_LABELS[predicted_idx.item()]

def predict_cnn_svm(image_path, cnn_checkpoint="fer_model.pth", svm_model_path="svm_model.joblib"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(cnn_checkpoint, weights_only=True))
    model.eval()

    svm_model = joblib.load(svm_model_path)

    img_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        features = model(img_tensor, extract_features=True)
        features_np = features.cpu().numpy()

    predicted_idx = svm_model.predict(features_np)[0]
    return EMOTION_LABELS[predicted_idx]

def run_inference(image_path, mode='cnn_svm'):
    if mode == 'cnn':
        pred = predict_cnn(image_path)
    elif mode == 'cnn_svm':
        try:
            pred = predict_cnn_svm(image_path)
        except FileNotFoundError:
            print("SVM model not found. Falling back to CNN-only prediction.")
            pred = predict_cnn(image_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Predicted emotion: {pred}")
    return pred

def evaluate_svm_on_test(data_path, cnn_checkpoint="fer_model.pth", svm_model_path="svm_model.joblib"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(EMOTION_LABELS)

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(cnn_checkpoint, weights_only=True))
    model.eval()

    svm_model = joblib.load(svm_model_path)

    _, _, test_loader = get_dataloaders(data_path, batch_size=8)

    all_preds, all_trues = [], []

    print("=" * 60)
    print(f"{'ID':<5} | {'True Label':<14} | {'Pred Label':<14} | Status")
    print("-" * 60)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = model(images, extract_features=True)
            feats_np = feats.cpu().numpy()
            preds = svm_model.predict(feats_np)

            for i in range(images.size(0)):
                true_idx = labels[i].item()
                pred_idx = preds[i]
                true_name = EMOTION_LABELS[true_idx]
                pred_name = EMOTION_LABELS[pred_idx]
                status = "\u2713" if true_idx == pred_idx else "\u2717"
                sample_id = len(all_trues) + i + 1
                print(f"{sample_id:02d}    | {true_name:<14} | {pred_name:<14} | {status}")

            all_preds.extend(preds)
            all_trues.extend(labels.numpy())

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_trues, all_preds):
        cm[t, p] += 1

    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = 100.0 * correct / total

    print("-" * 60)
    print(f"CNN+SVM Test Summary: Accuracy {accuracy:.2f}% ({correct}/{total})")
    print("=" * 60)

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(cax)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('CNN+SVM Confusion Matrix')
    thresh = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_normalized[i, j]
            ax.text(j, i, f'{val:.2f}',
                    ha="center", va="center",
                    color="white" if val > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix_svm.svg', format='svg')
    plt.close()
    print("Confusion Matrix saved as confusion_matrix_svm.svg")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Unified FER inference")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--mode', type=str, choices=['cnn', 'cnn_svm'], default='cnn_svm',
                        help='Inference mode: cnn (softmax) or cnn_svm (SVM on CNN features)')
    parser.add_argument('--cnn_checkpoint', type=str, default='fer_model.pth')
    parser.add_argument('--svm_model', type=str, default='svm_model.joblib')
    args = parser.parse_args()
    run_inference(args.image_path, args.mode)
