import torch
import cv2
import numpy as np
from model import EmotionCNN

def run_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load("fer_model.pth", weights_only=True))
    except FileNotFoundError:
        print("Error: Model weights not found. Please run training first.")
        return

    model.eval() 
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return
        
    img = cv2.resize(img, (48, 48))
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    
    with torch.no_grad(): 
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1) 
        
    prediction = emotion_labels[predicted_idx.item()]
    print(f"Inference complete. Predicted emotion: {prediction}")
