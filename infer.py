import torch
import cv2
import numpy as np
from model import EmotionCNN

def run_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型架構並載入權重
    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load("fer_model.pth", weights_only=True))
    except FileNotFoundError:
        print("Error: Model weights not found. Please run training first.")
        return

    model.eval() # 切換至評估模式
    
    # 2. 讀取與前處理影像 (模擬 DataLoader 的處理過程)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return
        
    img = cv2.resize(img, (48, 48))
    # 轉換為 NCHW 張量格式: (1, 1, 48, 48)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # 3. 執行推論
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
    
    with torch.no_grad(): # 關閉梯度追蹤以節省記憶體
        outputs = model(img_tensor)
        # 取得機率最大值的索引
        _, predicted_idx = torch.max(outputs, 1) 
        
    prediction = emotion_labels[predicted_idx.item()]
    print(f"Inference complete. Predicted emotion: {prediction}")
