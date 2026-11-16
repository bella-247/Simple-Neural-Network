from typing import Tuple
import numpy as np


def make_xor(
    n_samples: int = 200, noise: float = 0.08, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D XOR dataset across 4 cluster centers."""
    if seed is not None:
        np.random.seed(seed)

    k = n_samples // 4
    c00 = np.array([0.0, 0.0])
    c11 = np.array([1.0, 1.0])
    c01 = np.array([0.0, 1.0])
    c10 = np.array([1.0, 0.0])

    x00 = c00 + np.random.randn(k, 2) * noise
    x11 = c11 + np.random.randn(k, 2) * noise
    x01 = c01 + np.random.randn(k, 2) * noise
    x10 = c10 + np.random.randn(k, 2) * noise

    X = np.vstack([x00, x11, x01, x10])
    y = np.vstack([
        np.zeros((2 * k, 1)),
        np.ones((2 * k, 1)),
    ])

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def make_moons(
    n_samples: int = 200, noise: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving crescent half-circles."""
    if seed is not None:
        np.random.seed(seed)

    n_out = n_samples // 2
    n_in = n_samples - n_out

    theta_out = np.linspace(0, np.pi, n_out)
    x_out = np.cos(theta_out)
    y_out = np.sin(theta_out)

    theta_in = np.linspace(0, np.pi, n_in)
    x_in = 1.0 - np.cos(theta_in)
    y_in = 0.5 - np.sin(theta_in)

    X = np.vstack([
        np.column_stack([x_out, y_out]),
        np.column_stack([x_in, y_in]),
    ])
    X += np.random.randn(*X.shape) * noise

    y = np.vstack([
        np.zeros((n_out, 1)),
        np.ones((n_in, 1)),
    ])

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
