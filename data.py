"""
data.py — Synthetic Dataset Generators (XOR & Two Moons)
=========================================================

What is this?
-------------
This module generates 2D classification datasets using only pure NumPy.

Why these specific datasets?
----------------------------
A single linear layer (perceptron) can only draw a single straight line
as a decision boundary: w1*x1 + w2*x2 + b = 0.

Datasets like XOR and Two Moons are NOT linearly separable:
- There is no straight line in 2D space that can separate the two classes.
- A network MUST have at least one non-linear hidden layer (e.g., Sigmoid)
  to warp the input space and separate the classes successfully.

Functions:
----------
- make_xor()   : 4 Gaussian clusters forming an XOR pattern.
- make_moons() : 2 interlocking crescent moons (classic non-linear benchmark).
"""

from typing import Tuple
import numpy as np


def make_xor(
    n_samples: int = 200, noise: float = 0.08, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D continuous XOR dataset centered around 4 cluster centers.

    Cluster Centers:
    - (0, 0) -> Class 0
    - (1, 1) -> Class 0
    - (0, 1) -> Class 1
    - (1, 0) -> Class 1

    Parameters
    ----------
    n_samples : total number of samples to generate
    noise     : standard deviation of Gaussian noise added to coordinates
    seed      : random seed for reproducibility

    Returns
    -------
    X : coordinates, shape (n_samples, 2)
    y : binary labels, shape (n_samples, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    samples_per_cluster = n_samples // 4

    # 4 Centers
    c00 = np.array([0.0, 0.0])
    c11 = np.array([1.0, 1.0])
    c01 = np.array([0.0, 1.0])
    c10 = np.array([1.0, 0.0])

    # Generate points around centers
    x00 = c00 + np.random.randn(samples_per_cluster, 2) * noise
    x11 = c11 + np.random.randn(samples_per_cluster, 2) * noise
    x01 = c01 + np.random.randn(samples_per_cluster, 2) * noise
    x10 = c10 + np.random.randn(samples_per_cluster, 2) * noise

    # Combine coordinates
    X = np.vstack([x00, x11, x01, x10])

    # Class 0: (0,0) and (1,1); Class 1: (0,1) and (1,0)
    y = np.vstack([
        np.zeros((2 * samples_per_cluster, 1)),
        np.ones((2 * samples_per_cluster, 1)),
    ])

    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def make_moons(
    n_samples: int = 200, noise: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two interleaving crescent half-circles.

    Upper moon: centered around (0, 0) with radius 1
    Lower moon: shifted to (1, -0.5) with radius 1

    Parameters
    ----------
    n_samples : total number of points
    noise     : standard deviation of Gaussian noise added to points
    seed      : random seed for reproducibility

    Returns
    -------
    X : coordinates, shape (n_samples, 2)
    y : binary labels, shape (n_samples, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Upper moon (Class 0)
    theta_out = np.linspace(0, np.pi, n_samples_out)
    x_out = np.cos(theta_out)
    y_out = np.sin(theta_out)

    # Lower moon (Class 1, shifted)
    theta_in = np.linspace(0, np.pi, n_samples_in)
    x_in = 1.0 - np.cos(theta_in)
    y_in = 0.5 - np.sin(theta_in)

    # Stack coordinates
    X = np.vstack([
        np.column_stack([x_out, y_out]),
        np.column_stack([x_in, y_in]),
    ])

    # Add Gaussian jitter
    X += np.random.randn(*X.shape) * noise

    # Labels
    y = np.vstack([
        np.zeros((n_samples_out, 1)),
        np.ones((n_samples_in, 1)),
    ])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
