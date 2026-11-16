"""
grad_check.py — Numerical Gradient Verification Engine
======================================================

What is this?
-------------
A gradient checker mathematically proves that our backpropagation code
is 100% correct by comparing backpropagation gradients against the
definition of the derivative:

              f(θ + ε) - f(θ - ε)
    f'(θ) ≈ ───────────────────────
                    2ε

This is the two-sided finite difference approximation.

Why two-sided instead of one-sided?
------------------------------------
- One-sided: (f(θ + ε) - f(θ)) / ε       has truncation error O(ε)
- Two-sided: (f(θ + ε) - f(θ - ε)) / (2ε) has truncation error O(ε²)
  With ε = 1e-5, two-sided error is ~1e-10, making it vastly more precise.

Relative Error Formula:
-----------------------
Direct absolute difference |g_backprop - g_num| is misleading when gradients
are tiny or huge. We compute the normalized relative error:

                           || g_backprop - g_num ||₂
    relative_error = ─────────────────────────────────────────
                      || g_backprop ||₂ + || g_num ||₂ + 1e-8

Grading Bar:
------------
- relative error < 1e-6 : ✅ PASS (mathematical perfection)
- relative error < 1e-4 : ⚠️  SUSPICIOUS (likely a formulation bug)
- relative error >= 1e-4: ❌ FAIL (gradient math is broken)
"""

from typing import Dict, Any, Tuple
import numpy as np
from nn import Network, Linear, Sigmoid, MSELoss


def gradient_check(
    network: Network,
    criterion: Any,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5,
    threshold: float = 1e-6,
) -> bool:
    """
    Compare analytical backprop gradients to two-sided numerical gradients.

    Parameters
    ----------
    network   : instantiated Network model
    criterion : loss function instance (e.g., MSELoss)
    X         : input data array
    y         : target label array
    epsilon   : tiny perturbation step size (default: 1e-5)
    threshold : tolerance for passing (default: 1e-6)

    Returns
    -------
    passed : True if all parameter relative errors < threshold
    """
    # 1. Run standard forward & backward pass to compute analytical gradients
    y_pred = network.forward(X)
    loss = criterion.forward(y_pred, y)
    loss_grad = criterion.backward()
    network.backward(loss_grad)

    all_passed = True
    print("=" * 80)
    print(f"{'Layer':<10} | {'Param':<6} | {'Index':<8} | {'Analytical':<14} | {'Numerical':<14} | {'Rel Error':<12} | Status")
    print("-" * 80)

    # 2. Iterate through each trainable layer and parameter
    for layer_idx, layer in enumerate(network.layers):
        if not hasattr(layer, "W"):
            continue

        param_configs = [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]

        for name, param, grad in param_configs:
            # Flatten parameter array to iterate through each individual scalar
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index

                # Store original parameter value
                original_val = param[idx]

                # Perturb parameter: θ + ε
                param[idx] = original_val + epsilon
                loss_plus = criterion.forward(network.forward(X), y)

                # Perturb parameter: θ - ε
                param[idx] = original_val - epsilon
                loss_minus = criterion.forward(network.forward(X), y)

                # Restore original parameter value
                param[idx] = original_val

                # Two-sided numerical gradient
                num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                ana_grad = grad[idx]

                # Relative error
                diff = abs(ana_grad - num_grad)
                denom = max(1e-8, abs(ana_grad) + abs(num_grad))
                rel_error = diff / denom

                passed = rel_error < threshold
                if not passed:
                    all_passed = False

                status_str = "✅ PASS" if passed else "❌ FAIL"
                idx_str = str(idx)
                print(
                    f"Layer {layer_idx:<4} | {name:<6} | {idx_str:<8} | "
                    f"{ana_grad:<14.8f} | {num_grad:<14.8f} | {rel_error:<12.3e} | {status_str}"
                )

                it.iternext()

    print("=" * 80)
    if all_passed:
        print(f"🎉 ALL GRADIENTS PASSED! (All relative errors are strictly < {threshold})")
    else:
        print(f"⚠️  SOME GRADIENTS FAILED TO MEET THE {threshold} THRESHOLD.")

    return all_passed


def run_check():
    print("=" * 80)
    print("        PHASE 3: NUMERICAL GRADIENT CHECK FOR ALL PARAMETERS")
    print("=" * 80)

    # Create network with arbitrary random weights
    np.random.seed(42)
    net = Network([
        Linear(in_features=2, out_features=2),
        Sigmoid(),
        Linear(in_features=2, out_features=1),
    ])

    criterion = MSELoss()

    # Small batch of samples
    X = np.random.randn(3, 2)
    y = np.array([[1.0], [0.0], [1.0]])

    passed = gradient_check(net, criterion, X, y, epsilon=1e-5, threshold=1e-6)
    assert passed, "Gradient check failed!"


if __name__ == "__main__":
    run_check()

