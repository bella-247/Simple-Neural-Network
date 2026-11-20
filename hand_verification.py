import numpy as np
from nn import Network, Linear, Sigmoid, MSELoss


def run_hand_verification():
    print("=" * 80)
    print(" 🧠 PHASE 1: HAND VERIFICATION SCRIPT (4 SAMPLES)")
    print("=" * 80)

    # 1. Fixed Architecture & Parameters
    layer1 = Linear(in_features=2, out_features=2)
    sig = Sigmoid()
    layer2 = Linear(in_features=2, out_features=1)
    criterion = MSELoss()

    layer1.W = np.array([[0.1, 0.4], [0.3, 0.2]], dtype=float)
    layer1.b = np.array([0.1, 0.1], dtype=float)

    layer2.W = np.array([[0.5, 0.6]], dtype=float)
    layer2.b = np.array([0.2], dtype=float)

    net = Network([layer1, sig, layer2])

    # 2. The 4 Requested Inputs and their XOR Targets
    X = np.array([
        [0.0, 0.0],  # Sample 1 -> y = 0
        [1.0, 1.0],  # Sample 2 -> y = 0
        [1.0, 0.0],  # Sample 3 -> y = 1
        [0.1, 0.9],  # Sample 4 -> y = 1
    ])
    y = np.array([[0.0], [0.0], [1.0], [1.0]])

    print("\n--- DATASET SAMPLES ---")
    for i in range(len(X)):
        print(f"Sample #{i+1}: x = {X[i]}, y = {y[i][0]}")

    print("\n" + "=" * 80)
    print(" ▶ FORWARD PASS PER SAMPLE (STEP-BY-STEP)")
    print("=" * 80)

    for i in range(len(X)):
        x_i = X[i:i+1]
        y_i = y[i:i+1]

        z1 = layer1.forward(x_i)
        a1 = sig.forward(z1)
        y_hat = layer2.forward(a1)
        loss = criterion.forward(y_hat, y_i)

        print(f"\n--- Sample #{i+1}: x = {x_i[0]} | Target y = {y_i[0,0]} ---")
        print("  1. Pre-activations z1 = [z1_0, z1_1]:")
        print(f"     z1_0 = {z1[0,0]:.8f}")
        print(f"     z1_1 = {z1[0,1]:.8f}")
        print("  2. Sigmoid Activations a1 = [a1_0, a1_1]:")
        print(f"     a1_0 = {a1[0,0]:.8f}")
        print(f"     a1_1 = {a1[0,1]:.8f}")
        print("  3. Output Prediction ŷ:")
        print(f"     ŷ    = {y_hat[0,0]:.8f}")
        print("  4. Loss L = (ŷ - y)²:")
        print(f"     Loss = {loss:.8f}")

    print("\n" + "=" * 80)
    print(" ▶ BATCH FORWARD PASS (N = 4)")
    print("=" * 80)
    batch_preds = net.forward(X)
    batch_loss = criterion.forward(batch_preds, y)
    print(f"Batch Predictions ŷ:\n{batch_preds}")
    print(f"Mean MSE Loss: {batch_loss:.8f}")
    print("=" * 80)


if __name__ == "__main__":
    run_hand_verification()
