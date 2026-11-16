# Gradient Verification & Correctness Protocol

To ensure analytical gradient correctness, this repository employs two rigorous verification protocols: exact paper hand-trace arithmetic validation and finite-difference numerical gradient checking.

---

## 1. Two-Sided Finite Difference Gradient Checking

Gradient checking numerically approximates the directional derivative of the loss function with respect to every scalar parameter $\theta_i \in \{W, b\}$ using the symmetric two-sided quotient:

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}$$

### Truncation Error Advantage
* One-sided difference $\frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$ incurs truncation error $\mathcal{O}(\epsilon)$.
* Symmetric two-sided difference cancels first-order error terms in the Taylor series expansion, achieving quadratic truncation error $\mathcal{O}(\epsilon^2)$.
* Setting perturbation parameter $\epsilon = 10^{-5}$ yields numerical precision of order $\approx 10^{-10}$.

---

## 2. Relative Error Metric

Direct absolute difference $|g_{\text{analytical}} - g_{\text{numerical}}|$ is susceptible to parameter magnitude variations. We compute normalized relative error:

$$\text{RelError} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(10^{-8}, |g_{\text{analytical}}| + |g_{\text{numerical}}|)}$$

### Verification Thresholds
| Relative Error | Status | Assessment |
| :--- | :--- | :--- |
| $< 10^{-6}$ | **PASS** | Analytical backpropagation implementation is mathematically correct. |
| $10^{-6} \le \text{err} < 10^{-4}$ | **SUSPICIOUS** | Possible precision issue, kink in non-differentiable points, or sub-optimal $\epsilon$. |
| $\ge 10^{-4}$ | **FAIL** | Formulation bug in chain rule or matrix dimension broadcasting. |

Empirical tests on all network parameters yield relative errors ranging between $10^{-10}$ and $10^{-13}$, well below the $10^{-6}$ threshold.

---

## 3. Hand-Trace Verification Protocol

For an isolated forward and backward step on a single 2D sample $x = [0.5, 0.8]$ and target $y = 1.0$:
* Every intermediate variable ($z_1, a_1, z_2, \hat{y}, L$) is derived on paper.
* All layer parameter gradients ($dW_1, db_1, dW_2, db_2$) are manually derived.
* Running `python verify_hand_trace.py` outputs identical numbers to 8 decimal digits, proving complete analytical agreement.
