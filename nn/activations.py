import numpy as np


class Sigmoid:
    """Element-wise sigmoid activation: σ(z) = 1 / (1 + e^-z)."""

    def __init__(self):
        self.output = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-z))
        return self.output

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        local_grad = self.output * (1.0 - self.output)
        return upstream_grad * local_grad


class ReLU:
    """Element-wise rectified linear unit: f(z) = max(0, z)."""

    def __init__(self):
        self.input = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.input = z
        return np.maximum(0.0, z)

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        local_grad = (self.input > 0).astype(float)
        return upstream_grad * local_grad


class Tanh:
    """Element-wise hyperbolic tangent activation."""

    def __init__(self):
        self.output = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.output = np.tanh(z)
        return self.output

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        local_grad = 1.0 - self.output ** 2
        return upstream_grad * local_grad
