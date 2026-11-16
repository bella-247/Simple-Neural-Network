"""
trainer.py — Optimization and Training Loop
============================================

What is this?
-------------
The Trainer coordinates the learning process of the neural network.
It implements Stochastic Gradient Descent (SGD) to iteratively adjust
weights and biases to minimize the chosen loss function.

The 4-Step Training Cycle (Each Epoch)
--------------------------------------
For every iteration (epoch) over the dataset:

1. FORWARD PASS:
   Pass input X through the network to generate predictions ŷ.
   ŷ = network.forward(X)

2. COMPUTE LOSS:
   Evaluate how far predictions are from ground-truth labels y.
   loss = criterion.forward(ŷ, y)

3. BACKWARD PASS:
   Compute the gradient of loss w.r.t. all parameters via chain rule.
   dL_dy = criterion.backward()
   network.backward(dL_dy)

4. PARAMETER UPDATE (SGD):
   Move each parameter a small step in the opposite direction of its gradient.
   W = W - learning_rate * dW
   b = b - learning_rate * db
"""

from typing import List, Dict, Any
import numpy as np
from nn.network import Network


class Trainer:
    """
    Manages the optimization loop for a Network.

    Parameters
    ----------
    network       : instance of Network
    criterion     : loss function instance (e.g., MSELoss)
    learning_rate : step size for gradient descent (eta)
    """

    def __init__(self, network: Network, criterion: Any, learning_rate: float = 0.1):
        self.network = network
        self.criterion = criterion
        self.learning_rate = learning_rate

    def step_optimizer(self) -> None:
        """
        Update trainable parameters (W, b) across all layers using SGD.

        Rule:
            param = param - learning_rate * gradient
        """
        for layer in self.network.layers:
            # Check if this layer has learnable weights (e.g., Linear layer)
            if hasattr(layer, "W") and layer.dW is not None:
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        print_every: int = 100,
    ) -> List[float]:
        """
        Train the network on the provided dataset for a number of epochs.

        Parameters
        ----------
        X           : input features, shape (n_samples, in_features)
        y           : target labels, shape (n_samples, out_features)
        epochs      : total training iterations
        print_every : frequency (in epochs) to print current loss

        Returns
        -------
        loss_history : list of loss values per epoch
        """
        loss_history = []

        for epoch in range(1, epochs + 1):
            # 1. Forward pass
            y_pred = self.network.forward(X)

            # 2. Compute loss
            loss = self.criterion.forward(y_pred, y)
            loss_history.append(loss)

            # 3. Backward pass
            loss_grad = self.criterion.backward()
            self.network.backward(loss_grad)

            # 4. Update parameters
            self.step_optimizer()

            # Logging
            if epoch % print_every == 0 or epoch == 1:
                # Compute classification accuracy (predictions thresholded at 0.5)
                predictions = (y_pred >= 0.5).astype(int)
                accuracy = np.mean(predictions == y) * 100.0
                print(
                    f"Epoch {epoch:5d}/{epochs} | Loss: {loss:.6f} | Accuracy: {accuracy:.2f}%"
                )

        return loss_history
