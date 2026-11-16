import numpy as np


class Linear:
    """Fully connected affine layer: z = x @ W.T + b."""

    def __init__(self, in_features: int, out_features: int):
        # He/Xavier-style scaling for stable variance
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)

        self.input = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.W.T + self.b

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        # Gradients w.r.t parameters
        self.dW = upstream_grad.T @ self.input
        self.db = upstream_grad.sum(axis=0)

        # Gradient w.r.t layer input (passed to previous layer)
        return upstream_grad @ self.W
