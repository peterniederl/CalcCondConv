from tensorflow import keras
import numpy as np
from typing import Callable, Iterable, Tuple


class CosineAnnealingScheduler(keras.callbacks.Callback):
    """
    Cosine annealing learning rate scheduler with warmup.
    """
    def __init__(
        self,
        base_lr,
        min_lr,
        epochs,
        warmup_epochs=5,
        verbose=1,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        # Warmup phase
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine annealing phase
            p = (epoch - self.warmup_epochs) / max(
                1, self.epochs - self.warmup_epochs - 1
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * p)
            )

        learning_rate = self.model.optimizer.learning_rate
        if hasattr(learning_rate, "assign"):
            learning_rate.assign(lr)
        else:
            keras.backend.set_value(learning_rate, lr)

        if self.verbose and (epoch < 1 or (epoch + 1) % 5 == 0):
            phase = "warmup" if epoch < self.warmup_epochs else "cosine"
            print(f"> [LR Scheduler] epoch {epoch+1}: lr={lr:.6f} ({phase})")


