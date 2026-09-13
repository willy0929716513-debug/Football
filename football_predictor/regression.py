"""Small, dependency-light (numpy-only) linear and logistic regression
implementations used to blend situational features with the Elo rating
difference. No scikit-learn needed: logistic regression is fit with a few
iterations of Newton's method (IRLS), which converges in single digits of
iterations for a design matrix this small; linear regression is closed-form
ridge regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


@dataclass
class LinearModel:
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - np.array(self.mean)) / np.array(self.std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._transform(X)
        return Xs @ np.array(self.coef) + self.intercept


@dataclass
class LogisticModel:
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - np.array(self.mean)) / np.array(self.std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._transform(X)
        z = Xs @ np.array(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def fit_ridge_linear(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> LinearModel:
    mean, std = _standardize_fit(X)
    Xs = (X - mean) / std
    n, d = Xs.shape
    design = np.hstack([np.ones((n, 1)), Xs])
    penalty = l2 * np.eye(d + 1)
    penalty[0, 0] = 0.0  # never regularize the intercept
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return LinearModel(mean=mean.tolist(), std=std.tolist(), coef=weights[1:].tolist(), intercept=float(weights[0]))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 25) -> LogisticModel:
    mean, std = _standardize_fit(X)
    Xs = (X - mean) / std
    n, d = Xs.shape
    design = np.hstack([np.ones((n, 1)), Xs])
    weights = np.zeros(d + 1)
    penalty = l2 * np.eye(d + 1)
    penalty[0, 0] = 0.0

    for _ in range(iterations):
        z = design @ weights
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (p - y) + penalty @ weights
        w_diag = np.clip(p * (1 - p), 1e-6, None)
        hessian = (design * w_diag[:, None]).T @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        weights -= step
        if np.max(np.abs(step)) < 1e-8:
            break

    return LogisticModel(mean=mean.tolist(), std=std.tolist(), coef=weights[1:].tolist(), intercept=float(weights[0]))
