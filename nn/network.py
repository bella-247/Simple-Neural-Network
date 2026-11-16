from typing import List, Any
import numpy as np


class Network:
    """Sequential container orchestrating forward execution and reversed backpropagation."""

    def __init__(self, layers: List[Any]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
