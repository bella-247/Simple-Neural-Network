"""
verify_hand_trace.py — Phase 1 & 2 Concrete Number Verification
================================================================

What is this?
-------------
This script sets up the exact "fixed tiny network" specified in Phase 1:
- 2 inputs
- 2 hidden neurons with Sigmoid
- 1 output neuron
- MSE loss

It fixes deterministic weights, biases, and a single training point.
It prints every forward intermediate value and every backward gradient
to 8 decimal places so you can compare them directly with your paper hand trace.

Presentation Role:
------------------
When your mentors ask:
"Did you trace the forward and backward passes by hand?"
You show your paper calculation and run this script. Every single number
matches to the last decimal place, proving complete mastery of the mathematics.
"""

import numpy as np
from nn import Network, Linear, Sigmoid, MSELoss


def run_hand_trace_verification():
    print("=" * 70)
    print("       PHASE 1: HAND-TRACE VERIFICATION ON A SINGLE SAMPLE")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Setup exact single sample and network architecture
    # -------------------------------------------------------------
    x = np.array([[0.5, 0.8]])        # 1 sample, 2 features
    y_true = np.array([[1.0]])        # ground truth target

    # Instantiate layers
    linear1 = Linear(in_features=2, out_features=2)
    sigmoid = Sigmoid()
    linear2 = Linear(in_features=2, out_features=1)

    # Override random weights with clean, fixed values
    # Layer 1 weights & bias (2 inputs -> 2 hidden neurons)
    linear1.W = np.array([
        [0.1, 0.4],   # Neuron 1 weights: [w11, w12]
        [0.3, 0.2]    # Neuron 2 weights: [w21, w22]
    ])
    linear1.b = np.array([0.1, 0.1])

    # Layer 2 weights & bias (2 hidden inputs -> 1 output neuron)
    linear2.W = np.array([
        [0.5, 0.6]    # Output neuron weights: [w31, w32]
    ])
    linear2.b = np.array([0.2])

    # Assemble network
    net = Network([linear1, sigmoid, linear2])
    criterion = MSELoss()

    # -------------------------------------------------------------
    # 2. Forward Pass: Step-by-Step
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("▶ FORWARD PASS STEP-BY-STEP")
    print("-" * 70)
    print(f"Input x:             {x[0].tolist()}")
    print(f"Target y_true:       {y_true[0].tolist()}")

    # Step 2.1: Linear 1
    z1 = linear1.forward(x)
    print(f"\n1. Pre-activation z1 (W1 @ x + b1):")
    print(f"   z1[0] = 0.1*0.5 + 0.4*0.8 + 0.1 = {z1[0, 0]:.8f}")
    print(f"   z1[1] = 0.3*0.5 + 0.2*0.8 + 0.1 = {z1[0, 1]:.8f}")

    # Step 2.2: Sigmoid
    a1 = sigmoid.forward(z1)
    print(f"\n2. Hidden activation a1 (Sigmoid(z1)):")
    print(f"   a1[0] = 1 / (1 + exp(-{z1[0, 0]:.4f})) = {a1[0, 0]:.8f}")
    print(f"   a1[1] = 1 / (1 + exp(-{z1[0, 1]:.4f})) = {a1[0, 1]:.8f}")

    # Step 2.3: Linear 2
    z2 = linear2.forward(a1)
    y_pred = z2
    print(f"\n3. Output prediction ŷ (W2 @ a1 + b2):")
    print(f"   ŷ     = 0.5*{a1[0, 0]:.4f} + 0.6*{a1[0, 1]:.4f} + 0.2 = {y_pred[0, 0]:.8f}")

    # Step 2.4: MSE Loss
    loss = criterion.forward(y_pred, y_true)
    print(f"\n4. MSE Loss ( (ŷ - y_true)^2 ):")
    print(f"   Loss  = ({y_pred[0, 0]:.6f} - 1.0)^2 = {loss:.8f}")

    # -------------------------------------------------------------
    # 3. Backward Pass: Step-by-Step (Chain Rule)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("◀ BACKWARD PASS STEP-BY-STEP (CHAIN RULE)")
    print("-" * 70)

    # Step 3.1: Loss gradient w.r.t prediction
    dL_dypred = criterion.backward()
    print(f"1. dL/dŷ (2 * (ŷ - y_true) / N):")
    print(f"   dL/dŷ  = 2 * ({y_pred[0, 0]:.6f} - 1.0) = {dL_dypred[0, 0]:.8f}")

    # Step 3.2: Linear 2 backward
    dL_da1 = linear2.backward(dL_dypred)
    print(f"\n2. Gradients at Linear 2 (Output Layer):")
    print(f"   dL/dW2 = dL/dŷ @ a1.T = [{linear2.dW[0, 0]:.8f}, {linear2.dW[0, 1]:.8f}]")
    print(f"   dL/db2 = dL/dŷ        = {linear2.db[0]:.8f}")
    print(f"   dL/da1 = dL/dŷ @ W2   = [{dL_da1[0, 0]:.8f}, {dL_da1[0, 1]:.8f}]")

    # Step 3.3: Sigmoid backward
    dL_dz1 = sigmoid.backward(dL_da1)
    print(f"\n3. Gradients through Sigmoid (Chain Rule: dL/da1 * a1 * (1 - a1)):")
    print(f"   dL/dz1[0] = {dL_da1[0, 0]:.6f} * ({a1[0, 0]:.4f} * (1 - {a1[0, 0]:.4f})) = {dL_dz1[0, 0]:.8f}")
    print(f"   dL/dz1[1] = {dL_da1[0, 1]:.6f} * ({a1[0, 1]:.4f} * (1 - {a1[0, 1]:.4f})) = {dL_dz1[0, 1]:.8f}")

    # Step 3.4: Linear 1 backward
    dL_dx = linear1.backward(dL_dz1)
    print(f"\n4. Gradients at Linear 1 (Hidden Layer):")
    print(f"   dL/dW1[0, 0] = dL/dz1[0] * x[0] = {linear1.dW[0, 0]:.8f}")
    print(f"   dL/dW1[0, 1] = dL/dz1[0] * x[1] = {linear1.dW[0, 1]:.8f}")
    print(f"   dL/dW1[1, 0] = dL/dz1[1] * x[0] = {linear1.dW[1, 0]:.8f}")
    print(f"   dL/dW1[1, 1] = dL/dz1[1] * x[1] = {linear1.dW[1, 1]:.8f}")
    print(f"   dL/db1       = [{linear1.db[0]:.8f}, {linear1.db[1]:.8f}]")
    print(f"   dL/dx (input)= [{dL_dx[0, 0]:.8f}, {dL_dx[0, 1]:.8f}]")

    print("\n" + "=" * 70)
    print("                PAPER TRACE SUMMARY CHEAT SHEET")
    print("=" * 70)
    print(f"{'Variable':<15} | {'Forward Value':<16} | {'Gradient (dL/dVar)':<20}")
    print("-" * 57)
    print(f"{'z1[0]':<15} | {z1[0, 0]:<16.8f} | {dL_dz1[0, 0]:<20.8f}")
    print(f"{'z1[1]':<15} | {z1[0, 1]:<16.8f} | {dL_dz1[0, 1]:<20.8f}")
    print(f"{'a1[0]':<15} | {a1[0, 0]:<16.8f} | {dL_da1[0, 0]:<20.8f}")
    print(f"{'a1[1]':<15} | {a1[0, 1]:<16.8f} | {dL_da1[0, 1]:<20.8f}")
    print(f"{'z2 (ŷ)':<15} | {y_pred[0, 0]:<16.8f} | {dL_dypred[0, 0]:<20.8f}")
    print(f"{'Loss':<15} | {loss:<16.8f} | {'1.00000000 (dL/dL)':<20}")
    print("-" * 57)
    print(f"{'W2[0, 0]':<15} | {linear2.W[0, 0]:<16.8f} | {linear2.dW[0, 0]:<20.8f}")
    print(f"{'W2[0, 1]':<15} | {linear2.W[0, 1]:<16.8f} | {linear2.dW[0, 1]:<20.8f}")
    print(f"{'b2[0]':<15} | {linear2.b[0]:<16.8f} | {linear2.db[0]:<20.8f}")
    print(f"{'W1[0, 0]':<15} | {linear1.W[0, 0]:<16.8f} | {linear1.dW[0, 0]:<20.8f}")
    print(f"{'W1[0, 1]':<15} | {linear1.W[0, 1]:<16.8f} | {linear1.dW[0, 1]:<20.8f}")
    print(f"{'W1[1, 0]':<15} | {linear1.W[1, 0]:<16.8f} | {linear1.dW[1, 0]:<20.8f}")
    print(f"{'W1[1, 1]':<15} | {linear1.W[1, 1]:<16.8f} | {linear1.dW[1, 1]:<20.8f}")
    print(f"{'b1[0]':<15} | {linear1.b[0]:<16.8f} | {linear1.db[0]:<20.8f}")
    print(f"{'b1[1]':<15} | {linear1.b[1]:<16.8f} | {linear1.db[1]:<20.8f}")
    print("=" * 70)


if __name__ == "__main__":
    run_hand_trace_verification()
