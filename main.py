import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import data
from nn import Network, Linear, Sigmoid, MSELoss
from trainer import Trainer


def plot_results(
    network: Network, X: np.ndarray, y: np.ndarray, loss_history: list, prefix: str = ""
) -> None:
    if not HAS_MATPLOTLIB:
        return

    os.makedirs("assets", exist_ok=True)
    tag = f"_{prefix}" if prefix else ""

    # Loss convergence
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color="#1f77b4", linewidth=2)
    plt.title("MSE Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"assets/loss{tag}.png", dpi=150)
    plt.close()

    # Decision boundary
    plt.figure(figsize=(7, 6))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = network.forward(grid).reshape(xx.shape)
    plt.contourf(xx, yy, preds, levels=20, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(label="ŷ")
    plt.contour(xx, yy, preds, levels=[0.5], colors="black", linewidths=2)

    y_flat = y.ravel()
    plt.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], color="blue", edgecolor="k", s=30, label="Class 0")
    plt.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], color="red", edgecolor="k", s=30, label="Class 1")

    plt.title("Decision Boundary (Threshold = 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"assets/decision_boundary{tag}.png", dpi=150)
    plt.close()


def main():
    np.random.seed(42)

    # 1. Dataset selection (default: xor, or pass 'moons')
    dataset_name = sys.argv[1].lower() if len(sys.argv) > 1 else "xor"
    if dataset_name == "moons":
        print("Dataset: Two Moons")
        X, y = data.make_moons(n_samples=300, noise=0.1, seed=42)
    else:
        print("Dataset: XOR")
        X, y = data.make_xor(n_samples=300, noise=0.08, seed=42)

    # 2. Network architecture: 2 inputs -> 2 hidden (Sigmoid) -> 1 output
    model = Network([
        Linear(in_features=2, out_features=2),
        Sigmoid(),
        Linear(in_features=2, out_features=1),
    ])

    # 3. Train
    criterion = MSELoss()
    trainer = Trainer(model, criterion, learning_rate=0.1)
    losses = trainer.fit(X, y, epochs=3000, print_every=200)

    # 4. Evaluation
    preds = (model.forward(X) >= 0.5).astype(int)
    acc = np.mean(preds == y) * 100.0
    print(f"\nFinal Accuracy: {acc:.2f}% | Final Loss: {losses[-1]:.6f}")

    # 5. Visualization (saves to assets/)
    plot_results(model, X, y, losses)


if __name__ == "__main__":
    main()
