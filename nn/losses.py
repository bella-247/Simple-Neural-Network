"""
nn/losses.py — Loss Functions
==============================

What is a loss function?
--------------------------
A loss function measures how wrong the network's prediction is.
It takes the network's output (ŷ) and the true label (y) and returns
a single number — the loss. Lower loss means better predictions.

The loss function is the TOP of the computation graph. Its backward()
produces the very first gradient — the seed from which all other gradients
grow as we propagate backward through the network.

Unlike layer activations, loss.backward() takes NO upstream gradient.
It IS the start of the upstream. It simply returns dL/dŷ.

The two losses
--------------

MSELoss (Mean Squared Error)
    Use when: the output is a continuous value (regression)
    Forward:  L = (1/n) · Σ (ŷᵢ - yᵢ)²
    Backward: dL/dŷ = (2/n) · (ŷ - y)

BCELoss (Binary Cross-Entropy)
    Use when: the output is a probability (binary classification)
    Requires: ŷ in (0, 1) — typically after a Sigmoid output layer
    Forward:  L = -(1/n) · Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
    Backward: dL/dŷ = (1/n) · [-(y/ŷ) + (1-y)/(1-ŷ)]

For this project we use MSELoss as specified in the capstone brief.
BCELoss is included for completeness and because it's the standard
choice for binary classification tasks in practice.
"""

import numpy as np


class MSELoss:
    """
    Mean Squared Error loss: L = mean((ŷ - y)²)

    Compares each predicted value to the true value, squares the difference
    (so sign doesn't matter and large errors are penalized more), then averages.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the MSE loss.

        Parameters
        ----------
        y_pred : network output,  shape (batch_size, 1) or (batch_size,)
        y_true : ground truth,    same shape as y_pred

        Returns
        -------
        loss : a single scalar — the average squared error
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_pred.shape[0]
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        """
        Compute dL/dŷ — the gradient of the loss w.r.t. the network's output.

        This is the first gradient computed. Everything else flows from this.

        Returns
        -------
        dL/dŷ : shape same as y_pred
                = 2 · (ŷ - y) / n
        """
        return 2.0 * (self.y_pred - self.y_true) / self.n


class BCELoss:
    """
    Binary Cross-Entropy loss: L = -mean(y·log(ŷ) + (1-y)·log(1-ŷ))

    Measures the distance between two probability distributions:
    the network's prediction and the true label (0 or 1).

    Requires y_pred to be in (0, 1) — apply Sigmoid before using this loss.
    """

    # Small constant to avoid log(0) which is undefined
    _EPS = 1e-12

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the BCE loss.

        Parameters
        ----------
        y_pred : predicted probabilities, shape (batch_size, 1), values in (0, 1)
        y_true : ground truth labels,     same shape, values in {0, 1}

        Returns
        -------
        loss : a single scalar — the average binary cross-entropy
        """
        self.y_pred = np.clip(y_pred, self._EPS, 1.0 - self._EPS)  # prevent log(0)
        self.y_true = y_true
        self.n = y_pred.shape[0]
        return float(
            -np.mean(y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        )

    def backward(self) -> np.ndarray:
        """
        Compute dL/dŷ for binary cross-entropy.

        Returns
        -------
        dL/dŷ : shape same as y_pred
                = (1/n) · [-(y/ŷ) + (1-y)/(1-ŷ)]
        """
        return (-(self.y_true / self.y_pred) + (1 - self.y_true) / (1 - self.y_pred)) / self.n
