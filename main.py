from utils.trainer import Trainer


if __name__ == "__main__":
    trainer = Trainer(
        epochs=15,
        batch_size=128,
        learning_rate=1e-3,
        results_dir="results",
    )
    metrics = trainer.run()
