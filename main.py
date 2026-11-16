"""
main.py — End-to-End Execution and Demonstration
=================================================

What is this?
-------------
The entry point script that connects all modules together:
1. Generates a non-linear dataset (XOR or Two Moons) using `data.py`.
2. Instantiates the exact capstone network architecture:
   - 2 Inputs
   - 2 Hidden Neurons with Sigmoid activation
   - 1 Linear Output Neuron
3. Computes MSE loss using `MSELoss`.
4. Trains the network using `Trainer` with backpropagation and SGD.
5. Visualizes training progress:
   - Loss convergence curve over epochs.
   - 2D decision boundary separating the classes.
"""

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
    network: Network,
    X: np.ndarray,
    y: np.ndarray,
    loss_history: list,
    output_prefix: str = "xor",
) -> None:
    """
    Plot and save the loss curve and decision boundary (if matplotlib is installed).
    """
    if not HAS_MATPLOTLIB:
        print("\n[Notice] matplotlib is not installed. Skipping plot image generation.")
        print("To generate visual plots, install matplotlib: pip install matplotlib")
        return

    # 1. Plot Loss Curve
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color="#1f77b4", linewidth=2)
    plt.title("MSE Loss Convergence over Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    loss_filename = f"{output_prefix}_loss.png"
    plt.savefig(loss_filename, dpi=150)
    plt.close()
    print(f"Saved loss curve to {loss_filename}")

    # 2. Plot Decision Boundary
    plt.figure(figsize=(7, 6))

    # Create dense coordinate mesh grid across input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass across entire grid
    preds = network.forward(grid_points)
    preds = preds.reshape(xx.shape)

    # Contour filled background representing model prediction surface
    plt.contourf(xx, yy, preds, levels=20, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(label="Network Output (ŷ)")

    # Draw the decision boundary contour line at threshold 0.5
    plt.contour(xx, yy, preds, levels=[0.5], colors="black", linewidths=2)

    # Scatter actual training data points colored by class
    y_flat = y.ravel()
    plt.scatter(
        X[y_flat == 0, 0],
        X[y_flat == 0, 1],
        color="blue",
        edgecolor="k",
        s=40,
        label="Class 0",
    )
    plt.scatter(
        X[y_flat == 1, 0],
        X[y_flat == 1, 1],
        color="red",
        edgecolor="k",
        s=40,
        label="Class 1",
    )

    plt.title("Learned Decision Boundary (Threshold = 0.5)", fontsize=14, fontweight="bold")
    plt.xlabel("Feature $x_1$", fontsize=12)
    plt.ylabel("Feature $x_2$", fontsize=12)
    plt.legend(loc="upper right")
    plt.tight_layout()

    boundary_filename = f"{output_prefix}_decision_boundary.png"
    plt.savefig(boundary_filename, dpi=150)
    plt.close()
    print(f"Saved decision boundary to {boundary_filename}")


def run_demo():
    print("=" * 60)
    print("  CAPSTONE: INSIDE THE BACKWARD PASS — MODEL DEMO")
    print("=" * 60)

    # Set seed for reproducible initialization & dataset
    np.random.seed(42)

    # 1. Generate Non-Linear Dataset (XOR)
    print("\n[Step 1] Generating XOR dataset...")
    X, y = data.make_xor(n_samples=300, noise=0.08, seed=42)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # 2. Build Network Architecture as specified:
    # 2 inputs -> 2 hidden neurons with Sigmoid -> 1 output
    print("\n[Step 2] Building architecture: 2 -> [2 Sigmoid] -> 1 Linear...")
    network = Network([
        Linear(in_features=2, out_features=2),
        Sigmoid(),
        Linear(in_features=2, out_features=1),
    ])

    # 3. Setup Loss and Trainer
    criterion = MSELoss()
    learning_rate = 0.1
    epochs = 3000
    trainer = Trainer(network, criterion, learning_rate=learning_rate)

    # 4. Train the Model
    print(f"\n[Step 3] Training with SGD (epochs={epochs}, lr={learning_rate})...")
    losses = trainer.fit(X, y, epochs=epochs, print_every=500)

    # 5. Evaluate Final Accuracy
    final_preds = network.forward(X)
    binary_preds = (final_preds >= 0.5).astype(int)
    final_acc = np.mean(binary_preds == y) * 100.0
    print(f"\nFinal Training Accuracy: {final_acc:.2f}%")
    print(f"Initial Loss: {losses[0]:.6f} --> Final Loss: {losses[-1]:.6f}")

    # 6. Visualize and Save Plots
    print("\n[Step 4] Generating visualization plots...")
    plot_results(network, X, y, losses, output_prefix="xor")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
