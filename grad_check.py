from typing import Any
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
    Verify backprop gradients against two-sided numerical finite differences:
    f'(θ) ≈ (f(θ + ε) - f(θ - ε)) / (2ε)
    """
    # Analytical gradients via backpropagation
    y_pred = network.forward(X)
    criterion.forward(y_pred, y)
    network.backward(criterion.backward())

    all_passed = True
    print("=" * 90)
    print(f"{'Layer':<10} | {'Param':<6} | {'Index':<8} | {'Analytical':<14} | {'Numerical':<14} | {'Rel Error':<12} | Status")
    print("-" * 90)

    for layer_idx, layer in enumerate(network.layers):
        if not hasattr(layer, "W"):
            continue

        for name, param, grad in [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]:
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]

                # Two-sided numerical approximation
                param[idx] = orig + epsilon
                loss_plus = criterion.forward(network.forward(X), y)

                param[idx] = orig - epsilon
                loss_minus = criterion.forward(network.forward(X), y)

                param[idx] = orig

                num_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                ana_grad = grad[idx]

                denom = max(1e-8, abs(ana_grad) + abs(num_grad))
                rel_error = abs(ana_grad - num_grad) / denom

                passed = rel_error < threshold
                if not passed:
                    all_passed = False

                status = "PASS" if passed else "FAIL"
                print(
                    f"Layer {layer_idx:<4} | {name:<6} | {str(idx):<8} | "
                    f"{ana_grad:<14.8f} | {num_grad:<14.8f} | {rel_error:<12.3e} | {status}"
                )
                it.iternext()

    print("=" * 90)
    return all_passed


if __name__ == "__main__":
    np.random.seed(42)
    net = Network([
        Linear(in_features=2, out_features=2),
        Sigmoid(),
        Linear(in_features=2, out_features=1),
    ])
    loss_fn = MSELoss()

    X = np.random.randn(3, 2)
    y = np.array([[1.0], [0.0], [1.0]])

    passed = gradient_check(net, loss_fn, X, y)
    print("Gradient Check Result:", "ALL PASSED" if passed else "FAILED")
