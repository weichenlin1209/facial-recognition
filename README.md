# Facial Emotion Recognition (FER)

A complete pipeline for facial expression recognition: CLAHE histogram equalization, CNN feature extraction, and SVM classification.

## Pipeline Architecture

```
Input (48x48 grayscale)
    ↓
CLAHE (contrast enhancement)
    ↓
DataLoader (augmentation for training)
    ↓
CNN (3 conv blocks: 32→64→128, fc1=128)
    ↓
Two output paths:
  ├─ fc2 → 8-class softmax (CNN-only inference)
  └─ 128-d features → SVM (CNN+SVM inference)
```

## Dataset

- **CK+ Dataset** (`dataset/CK+ Dataset.csv`), 920 samples, 48x48 grayscale
- 8 emotion classes: Anger(0), Disgust(1), Fear(2), Happy(3), Sad(4), Surprise(5), Neutral(6), Contempt(7)
- **Split**:
  - Last 3 per class → test set (24 total)
  - Remaining → stratified 80/20 → training (716) / validation (180)
- CLAHE (clipLimit=2.0, tileGridSize=8x8) applied to all samples on load

## Requirements

```
torch torchvision opencv-python pandas numpy scikit-learn joblib matplotlib pillow
```

```bash
pip install torch torchvision opencv-python pandas numpy scikit-learn joblib matplotlib pillow
```

## File Structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Dataset loading, CLAHE, augmentations, stratified train/val/test split |
| `model.py` | EmotionCNN: 3 conv blocks, fc1=128, forward(x, extract_features) |
| `train.py` | CNN training loop (CrossEntropy, Adam, StepLR, early stopping) |
| `test.py` | CNN-only evaluation on test set with confusion matrix |
| `infer.py` | CNN-only single-image inference |
| `train_svm.py` | SVM training via GridSearchCV on CNN features |
| `predict.py` | CNN+SVM evaluation on test set and single-image inference |
| `main.py` | Unified CLI entry point |

## Usage

All operations go through `main.py`:

### Train CNN

```bash
python main.py --mode train --epochs 200
```

Trains the CNN feature extractor. Saves best checkpoint to `fer_model.pth`.

### Test CNN

```bash
python main.py --mode test
```

Evaluates CNN softmax classifier on the 24-sample test set.

### CNN Single-Image Inference

```bash
python main.py --mode infer --image_path path/to/image.jpg
```

### Train SVM

```bash
python main.py --mode train_svm
```

Extracts 128-d features from the trained CNN, runs GridSearchCV (C, kernel, gamma), saves best model to `svm_model.joblib`.

### Test CNN+SVM

```bash
python main.py --mode test_svm
```

Evaluates the full CNN+SVM pipeline on the test set with per-sample output.

### CNN+SVM Single-Image Inference

```bash
python main.py --mode predict --image_path path/to/image.jpg
```

## Training Details

- **Optimizer**: Adam (lr=1e-4, weight_decay=5e-4)
- **Scheduler**: StepLR (step=15, gamma=0.5)
- **Loss**: CrossEntropyLoss
- **Dropout**: 0.5 on fc1
- **Early stopping**: patience=20 epochs on validation accuracy
- **Augmentation**: CLAHE + RandomAffine(translate/shear/scale) + RandomRotation(10) + ColorJitter + RandomHorizontalFlip
- **Class balancing**: WeightedRandomSampler (8 classes × 300 samples per epoch)

## SVM Details

- **Feature dimension**: 128 (from CNN fc1 layer)
- **GridSearchCV**: C ∈ [0.01, 0.1, 0.5, 1.0], kernel ∈ [linear, rbf], gamma ∈ [scale, auto]
- **SVM params**: class_weight='balanced', 3-fold CV
