from typing import List, Any
import numpy as np


class Network:
    """Sequential container orchestrating forward execution and reversed backpropagation."""

    def __init__(self, layers: List[Any]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Sequential execution: Layer 0 -> Layer N-1
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Reverse execution: Layer N-1 -> Layer 0 (chain rule)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
