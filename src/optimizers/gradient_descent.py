import numpy as np # type: ignore
from src.optimizers.optimizer import Optimizer

class GradientDescent(Optimizer):

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        super().__init__(max_iter=max_iter, tol=tol)
        self.learning_rate = learning_rate

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def optimize(
        self, X: np.ndarray, y: np.ndarray, fit_intercept: bool = True
    ):
        if fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        m, n = X.shape
        weights = np.zeros(n)
        self.history = []

        for iteration in range(self.max_iter):
            z = np.dot(X, weights)
            predictions = self._sigmoid(z)

            errors = predictions - y
            gradient = (1.0 / m) * np.dot(X.T, errors)

            weights -= self.learning_rate * gradient

            loss = -(1.0 / m) * np.sum(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.history.append(loss)

            if np.linalg.norm(gradient) < self.tol:
                break

        return weights