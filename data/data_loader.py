import os
import numpy as np
from tensorflow.keras.utils import to_categorical


class MNISTDataLoader:

    NUM_CLASSES  = 10
    IMAGE_SIZE   = 28
    INPUT_SHAPE  = (28, 28, 1)

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test  = None
        self.y_test  = None
        self._loaded = False

    def load(self, npz_path: str = None) -> "MNISTDataLoader":

        X_train, y_train, X_test, y_test = self._fetch(npz_path)

        self.X_train = self._preprocess_images(X_train)
        self.X_test  = self._preprocess_images(X_test)
        self.y_train = to_categorical(y_train, self.NUM_CLASSES)
        self.y_test  = to_categorical(y_test,  self.NUM_CLASSES)

        self._loaded = True
        print(f"[DataLoader] Train: {self.X_train.shape} | Test: {self.X_test.shape}")
        return self

    def _fetch(self, npz_path):
        # 1. Local npz file
        if npz_path and os.path.exists(npz_path):
            print(f"[DataLoader] Loading from {npz_path}")
            d = np.load(npz_path)
            return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

        # 2. Keras built-in
        try:
            from tensorflow.keras.datasets import mnist as _mnist
            (X_train, y_train), (X_test, y_test) = _mnist.load_data()
            print("[DataLoader] Loaded via Keras.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            print(f"[DataLoader] Keras download failed ({e}). Trying sklearn...")

        # 3. sklearn OpenML fallback
        from sklearn.datasets import fetch_openml
        print("[DataLoader] Fetching via sklearn/OpenML (first run ~1 min)...")
        ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = ds.data.reshape(-1, 28, 28).astype(np.uint8)
        y = ds.target.astype(np.uint8)
        return X[:60000], y[:60000], X[60000:], y[60000:]

    def _preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """Normalize to [0,1] and reshape to (N, 28, 28, 1)."""
        images = images.astype("float32") / 255.0
        return images.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

    def get_train(self):
        self._check_loaded()
        return self.X_train, self.y_train

    def get_test(self):
        self._check_loaded()
        return self.X_test, self.y_test

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() before accessing data.")

    def summary(self) -> None:
        self._check_loaded()
        print("=" * 40)
        print("  MNIST Dataset Summary")
        print("=" * 40)
        print(f"  Train samples : {self.X_train.shape[0]:,}")
        print(f"  Test  samples : {self.X_test.shape[0]:,}")
        print(f"  Input shape   : {self.INPUT_SHAPE}")
        print(f"  Num classes   : {self.NUM_CLASSES}")
        print(f"  Pixel range   : [{self.X_train.min():.1f}, {self.X_train.max():.1f}]")
        print("=" * 40)
