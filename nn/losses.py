import numpy as np


class MSELoss:
    """Mean Squared Error loss: L = mean((ŷ - y)²)."""

    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.n = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_pred.shape[0]
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * (self.y_pred - self.y_true) / self.n


class BCELoss:
    """Binary Cross-Entropy loss for binary classification probabilities."""

    _EPS = 1e-12

    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.n = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, self._EPS, 1.0 - self._EPS)
        self.y_true = y_true
        self.n = y_pred.shape[0]
        return float(
            -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        )

    def backward(self) -> np.ndarray:
        return (-(self.y_true / self.y_pred) + (1 - self.y_true) / (1 - self.y_pred)) / self.n
