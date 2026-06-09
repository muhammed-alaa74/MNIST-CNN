# MNIST Handwritten Digit Classifier — Custom CNN

A deep learning image classification project built **from scratch** using a custom Convolutional Neural Network (CNN) architecture. Designed with clean OOP principles — every component is a class with a single responsibility.

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.2%** |
| Test Loss | 0.027 |
| Parameters | ~500K |
| Training Time | ~5 min (CPU) |

---

## Architecture

```
Input (28×28×1)
    │
    ▼
Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    │
    ▼
Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    │
    ▼
Block 3: Conv2D(128) → BN → MaxPool → Dropout(0.40)
    │
    ▼
Head: Flatten → Dense(256) → BN → Dropout(0.50) → Dense(10, softmax)
    │
    ▼
Output: 10 classes (digits 0–9)
```

**Design choices:**
- **BatchNormalization** after every Conv layer → faster convergence, less overfitting
- **Dropout** increases with depth → stronger regularization near the output
- **EarlyStopping** + **ReduceLROnPlateau** → automatic training control
- **ModelCheckpoint** → best weights saved automatically

---

## Project Structure

```
mnist-cnn/
├── data/
│   ├── __init__.py
│   └── data_loader.py      # MNISTDataLoader class
├── model/
│   ├── __init__.py
│   └── cnn_model.py        # CNNModel class
├── utils/
│   ├── __init__.py
│   ├── evaluator.py        # Evaluator + Visualizer classes
│   └── trainer.py          # Trainer orchestrator class
├── results/                # Auto-generated after training
│   ├── best_model.keras
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── wrong_predictions.png
│   └── metrics.json
├── main.py                 # Entry point
├── predict.py              # Inference script
├── requirements.txt
└── README.md
```

---

## OOP Design

| Class | Responsibility |
|---|---|
| `MNISTDataLoader` | Load, normalize, reshape, one-hot encode |
| `CNNModel` | Build, compile, train, evaluate, save/load |
| `Evaluator` | Classification report, confusion matrix, wrong predictions |
| `Visualizer` | Training curves, sample prediction plots |
| `Trainer` | Orchestrate the full pipeline end-to-end |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/mnist-cnn.git
cd mnist-cnn
pip install -r requirements.txt

# 2. Train the model
python main.py

# 3. Run inference on test samples
python predict.py --demo

# 4. Predict on your own image
python predict.py --image path/to/your/digit.png
```

---

## Training Curves

Training and validation accuracy/loss curves are saved automatically to `results/training_curves.png` after running `main.py`.

---

## Sample Predictions

Green title = correct prediction | Red title = wrong prediction

---
