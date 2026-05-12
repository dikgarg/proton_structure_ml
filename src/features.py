"""
Feature engineering helpers for the F2 ML project.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_poly_pipeline(degree=2):
    """Return a Pipeline: StandardScaler → PolynomialFeatures(degree)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ])


def add_physics_features(X):
    """Augment a (log10_x, log10_Q2) matrix with interaction features.

    Returns a matrix with columns:
        log10_x, log10_Q2,
        log10_x * log10_Q2,   (cross term)
        log10_x ** 2,         (small-x curvature)
        log10_Q2 ** 2,        (Q2-scaling curvature)
    """
    lx = X[:, 0:1]
    lQ2 = X[:, 1:2]
    return np.hstack([lx, lQ2, lx * lQ2, lx ** 2, lQ2 ** 2])


def standardise(X_train, X_test=None):
    """Fit StandardScaler on X_train and transform both splits.

    Returns (X_train_scaled, X_test_scaled, scaler).
    If X_test is None, returns (X_train_scaled, None, scaler).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test) if X_test is not None else None
    return X_tr, X_te, scaler
