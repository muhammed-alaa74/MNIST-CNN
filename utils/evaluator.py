import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


DIGIT_LABELS = [str(i) for i in range(10)]


class Evaluator:
    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray):
        self._model   = model
        self._X_test  = X_test
        self._y_test  = y_test
        self._y_true  = np.argmax(y_test, axis=1)
        self._y_pred  = model.predict(X_test)

    def classification_report(self) -> str:
        report = classification_report(
            self._y_true, self._y_pred,
            target_names=[f"Digit {i}" for i in range(10)]
        )
        print("\n[Evaluator] Classification Report:")
        print(report)
        return report

    def confusion_matrix(self, save_path: str = "results/confusion_matrix.png") -> None:
        cm = confusion_matrix(self._y_true, self._y_pred)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=DIGIT_LABELS, yticklabels=DIGIT_LABELS,
            ax=ax, linewidths=0.5
        )
        ax.set_title("Confusion Matrix — MNIST CNN", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] Confusion matrix saved → {save_path}")

    def wrong_predictions(self, n: int = 12, save_path: str = "results/wrong_predictions.png") -> None:
        """Show a grid of misclassified samples."""
        wrong_idx = np.where(self._y_true != self._y_pred)[0]
        sample    = wrong_idx[:n]

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle(f"Misclassified Samples (showing {len(sample)})", fontsize=14, fontweight="bold")

        for ax, idx in zip(axes.flat, sample):
            ax.imshow(self._X_test[idx].squeeze(), cmap="gray")
            ax.set_title(f"True: {self._y_true[idx]}  |  Pred: {self._y_pred[idx]}", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] Wrong predictions saved → {save_path}")


class Visualizer:

    def __init__(self, history):
        self._history = history.history

    def plot_training_curves(self, save_path: str = "results/training_curves.png") -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        epochs = range(1, len(self._history["accuracy"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History — MNIST CNN", fontsize=14, fontweight="bold")

        # Accuracy
        ax1.plot(epochs, self._history["accuracy"],     label="Train Accuracy", linewidth=2)
        ax1.plot(epochs, self._history["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
        ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
        ax1.legend(); ax1.grid(alpha=0.3)

        # Loss
        ax2.plot(epochs, self._history["loss"],     label="Train Loss", linewidth=2)
        ax2.plot(epochs, self._history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
        ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Training curves saved → {save_path}")

    def plot_sample_predictions(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n: int = 16,
        save_path: str = "results/sample_predictions.png"
    ) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        indices   = np.random.choice(len(X_test), n, replace=False)
        images    = X_test[indices]
        true_lbls = np.argmax(y_test[indices], axis=1)
        pred_lbls = model.predict(images)
        probas    = model.predict_proba(images)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle("Sample Predictions — MNIST CNN", fontsize=14, fontweight="bold")

        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].squeeze(), cmap="gray")
            conf  = probas[i][pred_lbls[i]] * 100
            color = "green" if pred_lbls[i] == true_lbls[i] else "red"
            ax.set_title(
                f"True: {true_lbls[i]}  Pred: {pred_lbls[i]}\nConf: {conf:.1f}%",
                fontsize=9, color=color
            )
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] Sample predictions saved → {save_path}")
