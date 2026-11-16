"""
nn/network.py — The Network (Sequential Orchestrator)
======================================================

What is this?
-------------
A neural network is simply a chain of layers connected in series:

    Input ---> [Layer 1] ---> [Layer 2] ---> ... ---> [Layer N] ---> Output

The Network class is the orchestrator. Its job is not to compute math
itself, but to organize the flow of data through all individual layers.

Why is this class essential?
----------------------------
1. Forward Pass (Prediction):
   Passes the input through each layer from left to right (in order).
   The output of layer `i` becomes the input to layer `i + 1`.

2. Backward Pass (Backpropagation):
   Passes the loss gradient through each layer in REVERSE order (right to left).
   Layer `i` takes the upstream gradient from layer `i + 1`, updates its own
   internal gradients (`dW`, `db`), and passes `dx` backward to layer `i - 1`.

   This reverse iteration IS the chain rule across the entire pipeline.
"""

from typing import List, Any
import numpy as np


class Network:
    """
    Sequential neural network container holding an ordered list of layers.

    Parameters
    ----------
    layers : list
        Ordered list of layer and activation instances
        (e.g., [Linear(2, 2), Sigmoid(), Linear(2, 1)]).
    """

    def __init__(self, layers: List[Any]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers in sequential order.

        Parameters
        ----------
        x : input data array of shape (batch_size, in_features)

        Returns
        -------
        out : network prediction after traversing all layers
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass through all layers in REVERSE order.

        Parameters
        ----------
        grad : initial gradient from the loss function (dL/dŷ)
               shape matching the network's final output

        Returns
        -------
        grad : gradient with respect to the original input (dL/dx)
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
