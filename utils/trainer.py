"""
utils/trainer.py
----------------
Trainer class — orchestrates the full pipeline:
    DataLoader → CNNModel → train → evaluate → visualize
"""

import json
import os
from data.data_loader import MNISTDataLoader
from model.cnn_model import CNNModel
from utils.evaluator import Evaluator, Visualizer


class Trainer:
    """
    Orchestrates the full ML pipeline.

    Usage:
        trainer = Trainer(epochs=15, batch_size=128)
        trainer.run()
    """

    def __init__(
        self,
        epochs: int        = 15,
        batch_size: int    = 128,
        learning_rate: float = 1e-3,
        results_dir: str   = "results",
    ):
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.results_dir   = results_dir

        self._loader  = MNISTDataLoader()
        self._model   = CNNModel(learning_rate=learning_rate)
        self._history = None
        self._metrics = None

    def run(self) -> dict:
        """Execute the full pipeline end-to-end."""
        print("\n" + "=" * 55)
        print("   MNIST CNN — Full Training Pipeline")
        print("=" * 55)

        # 1. Load data
        self._loader.load()
        self._loader.summary()
        X_train, y_train = self._loader.get_train()
        X_test,  y_test  = self._loader.get_test()

        # 2. Model summary
        print("\n[Trainer] Model Architecture:")
        self._model.summary()

        # 3. Train
        checkpoint = os.path.join(self.results_dir, "best_model.keras")
        self._history = self._model.train(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            checkpoint_path=checkpoint,
        )

        # 4. Evaluate
        self._metrics = self._model.evaluate(X_test, y_test)
        self._save_metrics()

        # 5. Visualize
        viz = Visualizer(self._history)
        viz.plot_training_curves(os.path.join(self.results_dir, "training_curves.png"))
        viz.plot_sample_predictions(
            self._model, X_test, y_test,
            save_path=os.path.join(self.results_dir, "sample_predictions.png")
        )

        # 6. Detailed evaluation
        evaluator = Evaluator(self._model, X_test, y_test)
        evaluator.classification_report()
        evaluator.confusion_matrix(os.path.join(self.results_dir, "confusion_matrix.png"))
        evaluator.wrong_predictions(os.path.join(self.results_dir, "wrong_predictions.png"))

        print("\n" + "=" * 55)
        print(f"   Pipeline complete.")
        print(f"   Test Accuracy : {self._metrics['test_accuracy']*100:.2f}%")
        print(f"   Results saved → ./{self.results_dir}/")
        print("=" * 55)

        return self._metrics

    def _save_metrics(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self._metrics, f, indent=2)
        print(f"[Trainer] Metrics saved → {path}")
