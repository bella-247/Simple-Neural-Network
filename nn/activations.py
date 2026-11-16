"""
nn/activations.py — Activation Functions
==========================================

What are activation functions?
--------------------------------
A Linear layer alone can only represent straight-line relationships.
Stack ten of them and you still have one big straight line — they collapse
into each other mathematically.

Activation functions fix this by introducing non-linearity after each
linear transformation. With non-linearity, hidden layers can learn curved
decision boundaries — the kind needed to separate XOR or two-moons data.

The pattern every activation follows
--------------------------------------
Every activation class has the same two methods:

    forward(z)              — applies the function element-wise
    backward(upstream_grad) — multiplies upstream gradient by local derivative

This "local derivative × upstream gradient" is the chain rule in action.
The local derivative tells us: "how sensitive is my output to my input?"

The three activations and their derivatives
--------------------------------------------

Sigmoid     σ(z) = 1 / (1 + e^-z)
  - Squashes any value into (0, 1)
  - Derivative: σ'(z) = σ(z) · (1 - σ(z))   ← beautiful: uses own output
  - Used in: hidden layers for binary problems, output for binary classification

ReLU        f(z) = max(0, z)
  - Passes positive values through unchanged, kills negatives
  - Derivative: 1 if z > 0, else 0           ← a simple gate
  - Used in: most modern deep networks (fast and effective)

Tanh        f(z) = (e^z - e^-z) / (e^z + e^-z)
  - Squashes any value into (-1, 1) — zero-centered unlike Sigmoid
  - Derivative: 1 - tanh²(z)                 ← uses own output
  - Used in: recurrent networks, when zero-centering matters
"""

import numpy as np


class Sigmoid:
    """
    Sigmoid activation: σ(z) = 1 / (1 + e^-z)

    Maps any real number to the range (0, 1).
    Its derivative is σ(z) · (1 - σ(z)) — computed from the saved output.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid element-wise.

        Parameters
        ----------
        z : pre-activation input, any shape

        Returns
        -------
        a : activated output, same shape as z, all values in (0, 1)
        """
        self.output = 1.0 / (1.0 + np.exp(-z))  # save output for backward
        return self.output

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Chain rule: upstream_grad · σ(z) · (1 - σ(z))

        Parameters
        ----------
        upstream_grad : gradient flowing back from the next layer

        Returns
        -------
        gradient to pass to the previous layer, same shape as upstream_grad
        """
        local_grad = self.output * (1.0 - self.output)
        return upstream_grad * local_grad


class ReLU:
    """
    ReLU activation: f(z) = max(0, z)

    Passes positive values unchanged and zeroes out negatives.
    Derivative is 1 where z > 0, and 0 everywhere else.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Apply ReLU element-wise.

        Parameters
        ----------
        z : pre-activation input, any shape

        Returns
        -------
        a : activated output, same shape, negatives replaced with 0
        """
        self.input = z  # save input — we need it to know where z was > 0
        return np.maximum(0.0, z)

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Chain rule: upstream_grad · (1 if z > 0 else 0)

        The gradient is simply blocked wherever the input was negative.

        Parameters
        ----------
        upstream_grad : gradient flowing back from the next layer

        Returns
        -------
        gradient to pass to the previous layer, same shape as upstream_grad
        """
        local_grad = (self.input > 0).astype(float)  # 1 where active, 0 where not
        return upstream_grad * local_grad


class Tanh:
    """
    Tanh activation: f(z) = (e^z - e^-z) / (e^z + e^-z)

    Maps any real number to (-1, 1). Zero-centered (unlike Sigmoid).
    Derivative is 1 - tanh²(z), computed from the saved output.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Apply tanh element-wise.

        Parameters
        ----------
        z : pre-activation input, any shape

        Returns
        -------
        a : activated output, same shape, all values in (-1, 1)
        """
        self.output = np.tanh(z)  # save output for backward
        return self.output

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Chain rule: upstream_grad · (1 - tanh²(z))

        Parameters
        ----------
        upstream_grad : gradient flowing back from the next layer

        Returns
        -------
        gradient to pass to the previous layer, same shape as upstream_grad
        """
        local_grad = 1.0 - self.output ** 2
        return upstream_grad * local_grad
