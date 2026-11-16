"""
nn/linear.py — The Linear Layer
================================

What is this?
-------------
A Linear layer performs the core computation of a neural network neuron:

    output = input @ W.T + b

Where:
    input  — the data coming in         shape: (batch_size, in_features)
    W      — the weight matrix          shape: (out_features, in_features)
    b      — the bias vector            shape: (out_features,)
    output — the result, pre-activation shape: (batch_size, out_features)

This is called "linear" because if you plot output vs input, every weight
combination produces a straight line. It is also called a "fully connected"
or "dense" layer because every input connects to every output neuron.

Why do we need forward AND backward?
--------------------------------------
- forward()  : computes the output during the prediction step
- backward() : computes three gradients during the learning step

The three gradients the backward pass computes:
    dW  — how much to change each weight      (same shape as W)
    db  — how much to change each bias        (same shape as b)
    dx  — gradient to pass back to prev layer (same shape as input)

How does backward know what to compute?
-----------------------------------------
It uses the chain rule. If the upstream gradient is dL/dz (how the loss
changes with respect to this layer's output z), then:

    dL/dW = dL/dz · dz/dW = upstream_grad.T @ input
    dL/db = dL/dz · dz/db = upstream_grad summed over the batch
    dL/dx = dL/dz · dz/dx = upstream_grad @ W

We store the input during forward() so backward() can use it.
"""

import numpy as np


class Linear:
    """
    A fully connected linear layer: output = input @ W.T + b

    Parameters
    ----------
    in_features  : number of inputs this layer receives
    out_features : number of outputs (neurons) this layer produces

    Attributes set after forward()
    --------------------------------
    input : the last input seen — stored so backward() can compute dW
    dW    : gradient of loss w.r.t. W — set by backward()
    db    : gradient of loss w.r.t. b — set by backward()
    """

    def __init__(self, in_features: int, out_features: int):
        # Xavier initialization: keeps signal variance stable across layers.
        # Without this, signals either explode or vanish as they pass through.
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)

        # Gradients — will be populated during backward()
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the linear transformation: z = x @ W.T + b

        Parameters
        ----------
        x : input array of shape (batch_size, in_features)

        Returns
        -------
        z : output array of shape (batch_size, out_features)
        """
        self.input = x  # save for use in backward()
        return x @ self.W.T + self.b

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Compute gradients using the chain rule.

        Parameters
        ----------
        upstream_grad : dL/dz — gradient flowing back from the next layer
                        shape: (batch_size, out_features)

        Returns
        -------
        dx : dL/dx — gradient to pass back to the previous layer
             shape: (batch_size, in_features)

        Side effects
        ------------
        Sets self.dW and self.db — the trainer reads these to update weights.
        """
        # dL/dW: how much each weight contributed to the loss
        # Shape: (out_features, in_features) — same as W
        self.dW = upstream_grad.T @ self.input

        # dL/db: bias gradient — sum over the batch dimension
        # Each sample in the batch adds its gradient to the bias
        self.db = upstream_grad.sum(axis=0)

        # dL/dx: gradient to pass upstream (to the layer before this one)
        # Shape: (batch_size, in_features) — same as input
        dx = upstream_grad @ self.W

        return dx
