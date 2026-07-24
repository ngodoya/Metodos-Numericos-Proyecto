import numpy as np # type: ignore
from src.optimizers.optimizer import Optimizer

class NewtonRaphson(Optimizer):

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        super().__init__(max_iter=max_iter, tol=tol)

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

            p = predictions
            w_diag = p * (1.0 - p)
            W = np.diag(w_diag)
            hessian = (1.0 / m) * np.dot(X.T, np.dot(W, X))

            hessian += np.eye(n) * 1e-6

            try:
                update = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                update = np.linalg.pinv(hessian) @ gradient

            weights -= update

            loss = -(1.0 / m) * np.sum(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.history.append(loss)

            if np.linalg.norm(gradient) < self.tol:
                break

        return weights