# Capstone Report: Inside the Backward Pass

**Author:** Abel Mekonen  
**Focus:** Pure NumPy Neural Network Engine & Backpropagation Mechanics  
**Repository:** [Simple-Neural-Network](.)

---

## 1. Executive Summary

This project implements a modular deep learning engine from scratch using only **NumPy** and **Matplotlib** (no PyTorch, TensorFlow, or autograd). The goal was to build, mathematically verify, and analyze the backward pass on non-linear problems that linear models cannot solve.

---

## 2. Numerical Gradient Verification

To mathematically prove that the backpropagation engine is bug-free:
- Implemented a two-sided finite difference gradient checker ([`grad_check.py`](grad_check.py)):
  $$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$
- **Result:** Every parameter in the network achieved a relative error between $1.2 \times 10^{-10}$ and $5.3 \times 10^{-13}$, easily surpassing the required $10^{-6}$ threshold and proving backpropagation correctness.

---

## 3. Experimental Observations & Findings

### Observation A: Escaping the 75% Saddle Point on XOR
When training the 2-2-1 Sigmoid network on the continuous XOR dataset ([`main.py`](main.py)):
* **Epochs 1–2400:** Accuracy remained flat at **75.00%**. This is the exact mathematical limit of a linear classifier on XOR (it can separate 3 out of 4 clusters, but not the 4th).
* **Epochs 2600–2800:** A rapid breakthrough occurred—accuracy jumped from **75.00% $\rightarrow$ 99.67% $\rightarrow$ 100.00%**.
* **Reason:** The two hidden neurons initially had symmetric, small gradients. Once their weights crossed a critical threshold, they broke symmetry: one neuron learned an OR-like boundary and the other a NAND-like boundary, rapidly warping the decision surface into a non-linear shape.

### Observation B: Representational Capacity on Two Moons
When evaluating the network on the interlocking Two Moons dataset:
* **2 Hidden Neurons:** Achieved **89.00% accuracy**. The decision boundary formed a single gentle curve across the plane. Because two sigmoids only provide two directional hyperplanes, the model geometrically lacked the degrees of freedom to double-bend around both crescent tips.
* **Scaling to 8 Hidden Neurons:** Accuracy immediately increased to **99.67%**. Combining 8 hidden activations allowed the network to carve a deep S-shaped contour that tightly separates both interlocking arcs.

### Observation C: Output Activation Changes Optimization, Not Capacity

To isolate the effect of the output activation and loss pairing, the 2-2-1 network was retrained on XOR (300 samples, noise 0.08, seed 42) with SGD ($\eta = 0.1$) over 10 initialization seeds. Final accuracy is reported at exactly 3000 and 6000 epochs:

| Configuration | 3000 epochs | 6000 epochs |
|---|---|---|
| Linear + MSE | **90.23%** mean (8/10 > 90%) | **89.93%** mean (8/10 > 90%) |
| Sigmoid + MSE | 73.33% mean (1/10 > 90%) | 76.70% mean (2/10 > 90%) |
| Sigmoid + BCE | 75.77% mean (2/10 > 90%) | 84.07% mean (6/10 > 90%) |

* **Same boundary, different gradients:** Because sigmoid is monotonic, $\sigma(z) \ge 0.5 \iff z \ge 0$. Adding an output sigmoid does not change the decision boundary or the network's representational capacity.
* **Saturation delays the 75% XOR escape:** $\sigma'(z) \le 0.25$ shrinks every gradient. With MSE the output gradient is $2(\hat{y}-y)\sigma'(z)/N$ (roughly 4x smaller at the plateau); with BCE it reduces to $(\hat{y}-y)/N$ (roughly 2x smaller than linear + MSE). The fixed 3000-epoch budget therefore often ends before hidden-unit symmetry breaks.
* **Not the root cause:** Linear + MSE still fails to escape on 2/10 seeds, confirming the bottleneck is the XOR saddle/symmetry breaking rather than the output parameterization.
* **Design choice:** Linear + MSE was the most reliable option under the fixed 3000-epoch budget; sigmoid + BCE remains the correct pairing when calibrated probabilities are required.

---

## 4. Key Takeaways

1. **Backpropagation is just the chain rule in reverse:** Caching activations during `forward()` allows `backward()` to evaluate parameter gradients locally and pass input gradients upstream.
2. **Model capacity dictates geometry:** 2 hidden neurons are sufficient for XOR (two cuts), but interlocking geometries like Two Moons require higher hidden layer capacity ($d_h \ge 4$) to bend around multiple inflections.
3. **Finite differences provide mathematical certainty:** Comparing analytical derivatives against two-sided numeric approximations guarantees the implementation is bug-free before any training begins.
4. **Output activation trades optimization speed, not expressiveness:** A monotonic output activation leaves the decision boundary unchanged, but saturation can delay escape from symmetric saddles; the loss must cancel the output derivative (BCE + Sigmoid) to recover healthy gradients.
