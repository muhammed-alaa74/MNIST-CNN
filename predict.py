import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from model.cnn_model import CNNModel
from data.data_loader import MNISTDataLoader


def preprocess_image(path: str) -> np.ndarray:
    """Load any image, convert to 28×28 grayscale, normalize."""
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


def run_demo():
    """Predict on 5 random MNIST test samples and print results."""
    loader = MNISTDataLoader().load()
    X_test, y_test = loader.get_test()

    model = CNNModel()
    model.load("results/best_model.keras")

    indices = np.random.choice(len(X_test), 5, replace=False)
    images  = X_test[indices]
    true    = np.argmax(y_test[indices], axis=1)
    preds   = model.predict(images)
    probas  = model.predict_proba(images)

    print("\n Demo Predictions:")
    print(f"{'#':<4} {'True':>6} {'Pred':>6} {'Confidence':>12}")
    print("-" * 32)
    for i, (t, p) in enumerate(zip(true, preds)):
        conf = probas[i][p] * 100
        status = "✓" if t == p else "✗"
        print(f"{i+1:<4} {t:>6} {p:>6} {conf:>11.1f}%  {status}")


def run_single(image_path: str):
    model = CNNModel()
    model.load("results/best_model.keras")
    image = preprocess_image(image_path)
    pred  = model.predict(image)[0]
    proba = model.predict_proba(image)[0]
    print(f"\nPredicted digit : {pred}")
    print(f"Confidence      : {proba[pred]*100:.2f}%")
    print(f"All probabilities:")
    for digit, p in enumerate(proba):
        bar = "█" * int(p * 30)
        print(f"  {digit}: {bar:<30} {p*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST CNN Predictor")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--demo",  action="store_true", help="Run demo on test set")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.image:
        run_single(args.image)
    else:
        print("Usage: python predict.py --demo")
        print("       python predict.py --image path/to/image.png")
