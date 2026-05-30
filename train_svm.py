import torch
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from model import EmotionCNN
from data_loader import get_dataloaders

EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']

def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model(images, extract_features=True)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def train_svm(data_path, cnn_checkpoint="fer_model.pth", svm_output="svm_model.joblib"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load(cnn_checkpoint, weights_only=True))
    except FileNotFoundError:
        print(f"Error: CNN checkpoint '{cnn_checkpoint}' not found. Train the CNN first.")
        return

    train_loader, _ = get_dataloaders(data_path, batch_size=32)

    print("Extracting CNN features for SVM training...")
    X_train, y_train = extract_features(model, train_loader, device)

    print(f"Train feature matrix: {X_train.shape}")

    param_grid = {
        'C': [0.01, 0.1, 0.5, 1.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }

    print("Running GridSearchCV for SVM hyperparameter tuning...")
    svc = SVC(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(
        svc, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    y_pred = grid_search.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    print(f"Training accuracy with best SVM: {train_acc:.4f}")
    print("\nClassification report (train):")
    print(classification_report(y_train, y_pred, target_names=EMOTION_LABELS))

    joblib.dump(grid_search.best_estimator_, svm_output)
    print(f"SVM model saved to '{svm_output}'")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train SVM on CNN features")
    parser.add_argument('--data_path', type=str, default='dataset/CK+ Dataset.csv')
    parser.add_argument('--cnn_checkpoint', type=str, default='fer_model.pth')
    parser.add_argument('--svm_output', type=str, default='svm_model.joblib')
    args = parser.parse_args()
    train_svm(args.data_path, args.cnn_checkpoint, args.svm_output)
