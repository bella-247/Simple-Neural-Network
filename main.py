from nn import BCELoss
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
from nn import Network, Linear, Sigmoid, MSELoss
from trainer import Trainer


def plot_results(
    network: Network, X: np.ndarray, y: np.ndarray, loss_history: list, prefix: str = "xor"
) -> None:
    os.makedirs("assets", exist_ok=True)
    loss_path = f"assets/{prefix}_loss.png"
    boundary_path = f"assets/{prefix}_decision_boundary.png"

    # Loss convergence
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color="#1f77b4", linewidth=2)
    plt.title(f"MSE Loss Convergence ({prefix.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
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

    plt.title(f"Decision Boundary - {prefix.upper()} (Threshold = 0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(boundary_path, dpi=150)
    plt.close()

    print(f"Saved plots to {loss_path} and {boundary_path}")


def main():
    np.random.seed(42)

    # 1. Dataset selection (default: xor, or pass 'moons')
    dataset_name = sys.argv[1].lower() if len(sys.argv) > 1 else "xor"
    if dataset_name == "moons":
        print("Dataset: Two Moons")
        X, y = data.make_moons(n_samples=300, noise=0.1, seed=42)
        test_X, test_Y = data.make_moons(n_samples=100, noise=0.1, seed=123)
    else:
        print("Dataset: XOR")
        X, y = data.make_xor(n_samples=300, noise=0.08, seed=42)
        test_X, test_Y = data.make_xor(n_samples=100, noise=0.08, seed=123)

    # 2. Network architecture: 2 inputs -> 2 hidden (Sigmoid) -> 1 output
    np.random.seed(42)
    model = Network([
        Linear(in_features=2, out_features=2),
        Sigmoid(),
        Linear(in_features=2, out_features=1),
    ])

    # 3. Train
    loss_fn = MSELoss()
    trainer = Trainer(model, loss_fn, learning_rate=0.1)
    losses = trainer.fit(X, y, epochs=3000, print_every=100)

    # 4. Evaluation (using unseen test data)
    preds = (model.forward(test_X) >= 0.5).astype(int)
    acc = np.mean(preds == test_Y) * 100.0
    test_loss = loss_fn.forward(model.forward(test_X), test_Y)
    print(f"\nTest Accuracy: {acc:.2f}% | Test Loss: {test_loss:.6f}")




    # 5. Print final trained parameters
    print("\n" + "=" * 55)
    print("🎯 Final Trained Parameters (Weights & Biases):")
    print("=" * 55)
    l1, l2 = model.layers[0], model.layers[2]
    print("Layer 1 (Hidden Layer):")
    print(f"  w11 = {l1.W[0, 0]:+.6f},  w12 = {l1.W[0, 1]:+.6f}")
    print(f"  w21 = {l1.W[1, 0]:+.6f},  w22 = {l1.W[1, 1]:+.6f}")
    print(f"  b1  = {l1.b[0]:+.6f},  b2  = {l1.b[1]:+.6f}")
    print("\nLayer 2 (Output Layer):")
    print(f"  w_out1 = {l2.W[0, 0]:+.6f},  w_out2 = {l2.W[0, 1]:+.6f}")
    print(f"  b_out  = {l2.b[0]:+.6f}")
    print("=" * 55)



    # 6. Visualization (saves with dataset prefix)
    plot_results(model, test_X, test_Y, losses, prefix=dataset_name)


if __name__ == "__main__":
    main()
