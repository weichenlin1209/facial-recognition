import argparse
from train import train_model
from infer import run_inference as run_cnn_inference
from test import evaluate_model
from train_svm import train_svm
from predict import run_inference as run_unified_inference
from predict import evaluate_svm_on_test

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition (FER) System")

    parser.add_argument('--mode', type=str,
                        choices=['train', 'test', 'infer', 'train_svm', 'test_svm', 'predict'],
                        required=True,
                        help='Mode: "train", "test" (CNN), "infer" (CNN single image), "train_svm", "test_svm" (CNN+SVM on test set), "predict" (CNN+SVM single image).')
    parser.add_argument('--data_path', type=str, default='dataset/CK+ Dataset.csv',
                        help='Path to the dataset CSV file.')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image for inference (required if mode is infer/predict).')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--cnn_checkpoint', type=str, default='fer_model.pth',
                        help='CNN checkpoint file (for train_svm/predict).')
    parser.add_argument('--svm_model', type=str, default='svm_model.joblib',
                        help='SVM model file (for predict).')

    args = parser.parse_args()

    if args.mode == 'train':
        print(f"Initiating training sequence with {args.epochs} epochs...")
        train_model(args.data_path, args.epochs)

    elif args.mode == 'test':
        evaluate_model(args.data_path)

    elif args.mode == 'infer':
        if not args.image_path:
            parser.error("--image_path is strictly required when mode is 'infer'")
        print(f"Initiating CNN inference on {args.image_path}...")
        run_cnn_inference(args.image_path)

    elif args.mode == 'train_svm':
        print("Training SVM on CNN features...")
        train_svm(args.data_path, args.cnn_checkpoint, args.svm_model)

    elif args.mode == 'test_svm':
        print("Evaluating CNN+SVM on test set...")
        evaluate_svm_on_test(args.data_path, args.cnn_checkpoint, args.svm_model)

    elif args.mode == 'predict':
        if not args.image_path:
            parser.error("--image_path is strictly required when mode is 'predict'")
        print(f"Running unified inference on {args.image_path}...")
        run_unified_inference(args.image_path, mode='cnn_svm')

if __name__ == '__main__':
    main()
