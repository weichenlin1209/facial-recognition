import argparse
from train import train_model
from infer import run_inference
from test import evaluate_model  # 新增此行匯入

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition (FER) System")
    
    # 將 'test' 加入 choices 陣列中
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'infer'], required=True,
                        help='Choose mode: "train", "test" (evaluate the 24 samples), or "infer" (single image).')
    parser.add_argument('--data_path', type=str, default='dataset/CK+ Dataset.csv',
                        help='Path to the dataset CSV file.')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image for inference (required if mode is infer).')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')

    args = parser.parse_args()

    # 控制流 (Control Flow) 擴充
    if args.mode == 'train':
        print(f"Initiating training sequence with {args.epochs} epochs...")
        train_model(args.data_path, args.epochs)
        
    elif args.mode == 'test':  # 新增的測試邏輯分支
        evaluate_model(args.data_path)
        
    elif args.mode == 'infer':
        if not args.image_path:
            parser.error("--image_path is strictly required when mode is 'infer'")
        print(f"Initiating inference sequence on {args.image_path}...")
        run_inference(args.image_path)

if __name__ == '__main__':
    main()
