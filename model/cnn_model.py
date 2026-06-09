import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CNNModel:

    def __init__(
        self,
        input_shape: tuple = (28, 28, 1),
        num_classes: int   = 10,
        learning_rate: float = 1e-3,
    ):
        self.input_shape   = input_shape
        self.num_classes   = num_classes
        self.learning_rate = learning_rate
        self._model: Model = self._build()

    # ── Architecture 
    def _build(self) -> Model:
        inputs = Input(shape=self.input_shape, name="input")

        # ── Block 1 ──
        x = Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
        x = BatchNormalization(name="bn1_1")(x)
        x = Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
        x = BatchNormalization(name="bn1_2")(x)
        x = MaxPooling2D((2, 2), name="pool1")(x)
        x = Dropout(0.25, name="drop1")(x)

        # ── Block 2 ──
        x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
        x = BatchNormalization(name="bn2_1")(x)
        x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
        x = BatchNormalization(name="bn2_2")(x)
        x = MaxPooling2D((2, 2), name="pool2")(x)
        x = Dropout(0.25, name="drop2")(x)

        # ── Block 3 ──
        x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
        x = BatchNormalization(name="bn3_1")(x)
        x = MaxPooling2D((2, 2), name="pool3")(x)
        x = Dropout(0.4, name="drop3")(x)

        # ── Head ──
        x = Flatten(name="flatten")(x)
        x = Dense(256, activation="relu", name="fc1")(x)
        x = BatchNormalization(name="bn_fc")(x)
        x = Dropout(0.5, name="drop_fc")(x)
        outputs = Dense(self.num_classes, activation="softmax", name="output")(x)

        model = Model(inputs=inputs, outputs=outputs, name="MNISTClassifier")
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Public API 
    def summary(self) -> None:
        self._model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int   = 20,
        batch_size: int = 128,
        validation_split: float = 0.1,
        checkpoint_path: str = "results/best_model.keras",
    ):

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-6, verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path, monitor="val_accuracy",
                save_best_only=True, verbose=1
            ),
        ]

        print(f"\n[CNNModel] Training for up to {epochs} epochs | batch={batch_size}")
        history = self._model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate on test set and return metrics dict."""
        loss, accuracy = self._model.evaluate(X_test, y_test, verbose=0)
        metrics = {"test_loss": round(loss, 4), "test_accuracy": round(accuracy, 4)}
        print(f"\n[CNNModel] Test Loss: {metrics['test_loss']} | Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
        return metrics

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return predicted class indices for a batch of images."""
        probs = self._model.predict(images, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Return raw softmax probabilities."""
        return self._model.predict(images, verbose=0)

    def save(self, path: str) -> None:
        self._model.save(path)
        print(f"[CNNModel] Model saved → {path}")

    def load(self, path: str) -> None:
        self._model = tf.keras.models.load_model(path)
        print(f"[CNNModel] Model loaded ← {path}")

    @property
    def keras_model(self) -> Model:
        return self._model
