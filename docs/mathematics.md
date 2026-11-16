# Mathematical Formulations

This document specifies the analytical definitions governing all forward transformations and backpropagation passes implemented in the engine.

---

## 1. Linear Affine Layer

Given an input batch matrix $X \in \mathbb{R}^{N \times d_{\text{in}}}$, weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, and bias vector $b \in \mathbb{R}^{d_{\text{out}}}$:

### Forward Transformation
$$z = X W^T + b \quad \in \mathbb{R}^{N \times d_{\text{out}}}$$

### Backward Gradients
Given upstream gradient $G = \frac{\partial L}{\partial z} \in \mathbb{R}^{N \times d_{\text{out}}}$:

$$\frac{\partial L}{\partial W} = G^T X \quad \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

$$\frac{\partial L}{\partial b} = \sum_{i=1}^N G_{i, :} \quad \in \mathbb{R}^{d_{\text{out}}}$$

$$\frac{\partial L}{\partial X} = G W \quad \in \mathbb{R}^{N \times d_{\text{in}}}$$

---

## 2. Activation Functions

### Sigmoid
* **Forward:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Backward Derivative:**
  $$\frac{d\sigma}{dz} = \sigma(z) \odot (1 - \sigma(z))$$

### Rectified Linear Unit (ReLU)
* **Forward:**
  $$f(z) = \max(0, z)$$
* **Backward Derivative:**
  $$\frac{df}{dz} = \mathbb{I}(z > 0)$$

### Hyperbolic Tangent (Tanh)
* **Forward:**
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
* **Backward Derivative:**
  $$\frac{d\tanh}{dz} = 1 - \tanh^2(z)$$

---

## 3. Loss Functions

### Mean Squared Error (MSE)
For prediction batch $\hat{y} \in \mathbb{R}^{N \times K}$ and ground truth $y \in \mathbb{R}^{N \times K}$:

* **Loss:**
  $$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|_2^2$$
* **Gradient:**
  $$\frac{\partial L_{\text{MSE}}}{\partial \hat{y}} = \frac{2}{N} (\hat{y} - y)$$

### Binary Cross-Entropy (BCE)
* **Loss:**
  $$L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
* **Gradient:**
  $$\frac{\partial L_{\text{BCE}}}{\partial \hat{y}} = \frac{1}{N} \left[ -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} \right]$$

---

## 4. Parameter Optimization

Parameters $\theta \in \{W, b\}$ are updated per epoch via standard first-order gradient descent:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \frac{\partial L}{\partial \theta}$$

where $\eta > 0$ represents the configured learning rate scalar.

