# System Architecture

## Overview

This repository implements a lightweight, modular deep learning framework constructed strictly using NumPy. The design follows an object-oriented paradigm analogous to modern autograd frameworks, decoupling forward execution, backpropagation, and parameter optimization into distinct layers and abstraction boundaries.

```
       [Input Data: X]
              │
              ▼
    ┌──────────────────┐
    │   Linear Layer   │ ◄── Affine Transformation (z = XWᵀ + b)
    └─────────┬────────┘
              ▼
    ┌──────────────────┐
    │ Activation Layer │ ◄── Non-Linear Mapping (e.g. Sigmoid, ReLU, Tanh)
    └─────────┬────────┘
              ▼
    ┌──────────────────┐
    │   Linear Layer   │ ◄── Projection to Output Dimension
    └─────────┬────────┘
              ▼
    ┌──────────────────┐
    │  Loss Criterion  │ ◄── Error Evaluation (e.g. MSELoss, BCELoss)
    └─────────┬────────┘
              │ (dL/dŷ)
              ▼ (Backpropagation Cascade)
```

---

## Core Components

### 1. `nn.Linear`
* **Role:** Fully-connected affine transformation layer.
* **State:** Maintains weight tensor `W` of shape `(out_features, in_features)` and bias vector `b` of shape `(out_features,)`.
* **Forward Contract:** Computes $z = XW^T + b$ and caches the incoming activation tensor $X$.
* **Backward Contract:** Consumes upstream gradient $\frac{\partial L}{\partial z}$, computes parameter gradients $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$, and returns the input gradient $\frac{\partial L}{\partial X}$.

### 2. `nn.activations` (`Sigmoid`, `ReLU`, `Tanh`)
* **Role:** Introduces non-linear functional representations.
* **Forward Contract:** Applies point-wise non-linear mappings. Caches pre-activations or post-activations required for derivative computation.
* **Backward Contract:** Multiplies upstream gradients by the local analytical derivative via element-wise (Hadamard) product.

### 3. `nn.losses` (`MSELoss`, `BCELoss`)
* **Role:** Evaluates discrepancy between model hypothesis $\hat{y}$ and empirical target $y$.
* **Backward Contract:** Serves as the root of the reverse computation graph, generating the seed gradient $\frac{\partial L}{\partial \hat{y}}$ without requiring incoming upstream derivatives.

### 4. `nn.Network`
* **Role:** Sequential directed acyclic graph (DAG) container.
* **Invariants:**
  * Forward traversal executes layers in index order: $0 \rightarrow N-1$.
  * Backward traversal executes layers in strict reverse index order: $N-1 \rightarrow 0$.

### 5. `trainer.Trainer`
* **Role:** Coordinates mini-batch iteration, gradient descent updates, and epoch metrics tracking.
* **Update Scheme:** Stochastic Gradient Descent (SGD) parameterized by learning rate $\eta$.
