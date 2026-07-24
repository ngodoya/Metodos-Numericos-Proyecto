from abc import ABC, abstractmethod
import numpy as np  # type: ignore


class Optimizer(ABC):

    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.history: list = []

    @abstractmethod
    def optimize(self, X: np.ndarray, y: np.ndarray, fit_intercept: bool = True):
        pass