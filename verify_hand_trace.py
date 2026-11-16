import numpy as np
from nn import Network, Linear, Sigmoid, MSELoss


def run_verification():
    x = np.array([[0.5, 0.8]])
    y_true = np.array([[1.0]])

    linear1 = Linear(in_features=2, out_features=2)
    sigmoid = Sigmoid()
    linear2 = Linear(in_features=2, out_features=1)

    # Fixed weights and biases for Phase 1 verification
    linear1.W = np.array([[0.1, 0.4], [0.3, 0.2]])
    linear1.b = np.array([0.1, 0.1])
    linear2.W = np.array([[0.5, 0.6]])
    linear2.b = np.array([0.2])

    net = Network([linear1, sigmoid, linear2])
    criterion = MSELoss()

    # Forward
    z1 = linear1.forward(x)
    a1 = sigmoid.forward(z1)
    y_pred = linear2.forward(a1)
    loss = criterion.forward(y_pred, y_true)

    # Backward
    dL_dypred = criterion.backward()
    dL_da1 = linear2.backward(dL_dypred)
    dL_dz1 = sigmoid.backward(dL_da1)
    linear1.backward(dL_dz1)

    print("=" * 65)
    print("           PHASE 1 HAND-TRACE VERIFICATION SUMMARY")
    print("=" * 65)
    print(f"{'Variable':<15} | {'Forward Value':<16} | {'Gradient (dL/dVar)':<20}")
    print("-" * 65)
    print(f"{'z1[0]':<15} | {z1[0, 0]:<16.8f} | {dL_dz1[0, 0]:<20.8f}")
    print(f"{'z1[1]':<15} | {z1[0, 1]:<16.8f} | {dL_dz1[0, 1]:<20.8f}")
    print(f"{'a1[0]':<15} | {a1[0, 0]:<16.8f} | {dL_da1[0, 0]:<20.8f}")
    print(f"{'a1[1]':<15} | {a1[0, 1]:<16.8f} | {dL_da1[0, 1]:<20.8f}")
    print(f"{'z2 (ŷ)':<15} | {y_pred[0, 0]:<16.8f} | {dL_dypred[0, 0]:<20.8f}")
    print(f"{'Loss':<15} | {loss:<16.8f} | {'1.00000000':<20}")
    print("-" * 65)
    print(f"{'W2[0, 0]':<15} | {linear2.W[0, 0]:<16.8f} | {linear2.dW[0, 0]:<20.8f}")
    print(f"{'W2[0, 1]':<15} | {linear2.W[0, 1]:<16.8f} | {linear2.dW[0, 1]:<20.8f}")
    print(f"{'b2[0]':<15} | {linear2.b[0]:<16.8f} | {linear2.db[0]:<20.8f}")
    print(f"{'W1[0, 0]':<15} | {linear1.W[0, 0]:<16.8f} | {linear1.dW[0, 0]:<20.8f}")
    print(f"{'W1[0, 1]':<15} | {linear1.W[0, 1]:<16.8f} | {linear1.dW[0, 1]:<20.8f}")
    print(f"{'W1[1, 0]':<15} | {linear1.W[1, 0]:<16.8f} | {linear1.dW[1, 0]:<20.8f}")
    print(f"{'W1[1, 1]':<15} | {linear1.W[1, 1]:<16.8f} | {linear1.dW[1, 1]:<20.8f}")
    print(f"{'b1[0]':<15} | {linear1.b[0]:<16.8f} | {linear1.db[0]:<20.8f}")
    print(f"{'b1[1]':<15} | {linear1.b[1]:<16.8f} | {linear1.db[1]:<20.8f}")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
