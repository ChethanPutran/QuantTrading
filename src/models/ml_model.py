from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseModel


class SklearnModel(BaseModel):
    """Adapter that exposes a scikit-learn estimator through BaseModel."""

    def __init__(self, estimator: Any, partial_fit: bool = False):
        self.estimator = estimator
        self.partial_fit = partial_fit

    def predict(self, x: np.ndarray) -> float:
        features = np.asarray(x, dtype=float).reshape(1, -1)
        prediction = self.estimator.predict(features)
        return float(np.asarray(prediction).reshape(-1)[0])

    def update(self, x: np.ndarray, y: float) -> None:
        if self.partial_fit and hasattr(self.estimator, "partial_fit"):
            self.estimator.partial_fit(
                np.asarray(x, dtype=float).reshape(1, -1),
                np.asarray([y], dtype=float),
            )
            return
        raise NotImplementedError(
            "wrapped estimator does not support online partial_fit updates"
        )


def get_model(model_name: str = "rtr", **kwargs: Any):
    """Create a common scikit-learn compatible estimator by short name."""

    name = model_name.lower()

    if name == "dtc":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(**kwargs)
    if name == "dtr":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(**kwargs)
    if name == "rtc":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**kwargs)
    if name == "rtr":
        from sklearn.ensemble import RandomForestRegressor

        defaults = {"n_estimators": 100, "random_state": 42}
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)
    if name == "svc":
        from sklearn.svm import SVC

        defaults = {"kernel": "rbf", "C": 1.0}
        defaults.update(kwargs)
        return SVC(**defaults)
    if name == "svr":
        from sklearn.svm import SVR

        defaults = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
        defaults.update(kwargs)
        return SVR(**defaults)
    if name == "lr":
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**kwargs)
    if name == "xgbc":
        from xgboost import XGBClassifier

        return XGBClassifier(**kwargs)
    if name == "xgbr":
        from xgboost import XGBRegressor

        defaults = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        defaults.update(kwargs)
        return XGBRegressor(**defaults)

    raise ValueError(f"unknown model_name: {model_name!r}")

