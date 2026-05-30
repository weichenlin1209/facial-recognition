import torch
import cv2
import numpy as np
import joblib
from model import EmotionCNN

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
