"""Shared helpers for selectable classical segmentation models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ..features import sliding_window_features


MODEL_CHOICES = ("svm", "logreg", "rf")


def build_model(name: str):
    """Create a supported classical model from a short name."""

    normalized = name.lower()
    if normalized == "svm":
        return SVC(kernel="rbf", probability=True)
    if normalized == "logreg":
        return LogisticRegression(max_iter=1000)
    if normalized == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_CHOICES)}")


def train_model(features: np.ndarray, labels: np.ndarray, name: str):
    """Fit a supported model on the provided feature matrix."""

    model = build_model(name)
    model.fit(features, labels)
    return model


def save_model(model, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as fh:
        pickle.dump(model, fh)


def load_model(source: Path):
    with open(source, "rb") as fh:
        return pickle.load(fh)


def predict_image(image: np.ndarray, model, window_size: int, return_probabilities: bool = False) -> np.ndarray:
    """Predict per-pixel labels from sliding-window features."""

    flat_features = sliding_window_features(image, window_size)
    if return_probabilities and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(flat_features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.reshape(-1)
    return model.predict(flat_features)
