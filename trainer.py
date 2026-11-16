from typing import List, Any
import numpy as np
from nn.network import Network


class Trainer:
    """Manages forward-backward execution and SGD parameter updates."""

    def __init__(self, network: Network, criterion: Any, learning_rate: float = 0.1):
        self.network = network
        self.criterion = criterion
        self.learning_rate = learning_rate

    def step_optimizer(self) -> None:
        """Update trainable parameters using standard SGD."""
        for layer in self.network.layers:
            if hasattr(layer, "W") and layer.dW is not None:
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        print_every: int = 100,
    ) -> List[float]:
        loss_history = []

        for epoch in range(1, epochs + 1):
            y_pred = self.network.forward(X)
            loss = self.criterion.forward(y_pred, y)
            loss_history.append(loss)

            loss_grad = self.criterion.backward()
            self.network.backward(loss_grad)

            self.step_optimizer()

            if epoch % print_every == 0 or epoch == 1:
                preds = (y_pred >= 0.5).astype(int)
                acc = np.mean(preds == y) * 100.0
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {loss:.6f} | Accuracy: {acc:.2f}%")

        return loss_history
