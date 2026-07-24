import numpy as np # type: ignore
from typing import Optional
from src.optimizers.optimizer import Optimizer


class LogisticRegression:

    def __init__(
        self, optimizer: Optimizer, fit_intercept: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.fit_intercept = fit_intercept
        self.weights = None
        self.intercept_: Optional[float] = None
        self.coef_ = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        raw_weights = self.optimizer.optimize(
            X, y, fit_intercept=self.fit_intercept
        )

        if self.fit_intercept:
            self.intercept_ = raw_weights[0]
            self.coef_ = raw_weights[1:]
            self.weights = raw_weights
        else:
            self.intercept_ = 0.0
            self.coef_ = raw_weights
            self.weights = raw_weights

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError(
                "El modelo no ha sido entrenado. Llama a .fit() primero."
            )

        X = np.asarray(X, dtype=np.float64)

        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        z = np.dot(X, self.weights)
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)