from __future__ import annotations


"""Ensemble decision engine combining model, regime, memory, and MPC signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.events import ActionEvent, StateVector, TradeAction
from ..memory.pattern import PatternRecord
from .base import ModelOutput

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseModel
from ..regime.gmm import BaseRegimeModel
from ..regime.hmm import BaseTransitionModel



InputTransform = Callable[[np.ndarray], Any]
OutputTransform = Callable[[Any], float]



@dataclass
class EnsembleDecisionEngine:
    action_threshold: float = 0.15

    def decide(
        self,
        state: StateVector,
        model_output: ModelOutput,
        regime_probs: np.ndarray,
        pattern: PatternRecord | None,
        mpc_forecast: np.ndarray,
    ) -> ActionEvent:
        action_scores = dict(model_output.action_logits)
        regime_bias = float(regime_probs[-1] if regime_probs.size else 0.0)
        pattern_bias = float(pattern.confidence if pattern is not None else 0.5)
        forecast_bias = float(np.mean(mpc_forecast)) if mpc_forecast.size else 0.0

        action_scores["BUY"] += max(0.0, model_output.predicted_return) + regime_bias + 0.25 * pattern_bias + 0.1 * forecast_bias
        action_scores["SELL"] += max(0.0, -model_output.predicted_return) + 0.25 * (1.0 - pattern_bias)
        action_scores["HOLD"] += model_output.uncertainty + 0.1 * (1.0 - abs(forecast_bias))

        action_name = max(action_scores, key=action_scores.get)
        confidence = float(np.clip(max(action_scores.values()) - min(action_scores.values()), 0.0, 1.0))
        if confidence < self.action_threshold:
            action_name = "HOLD"

        action = TradeAction(action_name)
        expected_reward = float(model_output.predicted_return)
        expected_risk = float(np.std(mpc_forecast) if mpc_forecast.size else model_output.uncertainty)
        return ActionEvent(
            state=state,
            action=action,
            confidence=confidence,
            expected_reward=expected_reward,
            expected_risk=expected_risk,
            planned_trajectory=np.asarray(mpc_forecast, dtype=float),
        )


@dataclass
class EnsembleMember:
    """One weighted model inside an ensemble."""

    name: str
    model: Any
    weight: float = 1.0
    input_transform: InputTransform | None = None
    output_transform: OutputTransform | None = None


class ModelEnsemble(BaseModel):
    """Weighted ensemble with a single ``predict`` method.

    Members can be project ``BaseModel`` instances, scikit-learn estimators,
    PyTorch/Keras-style models, or simple callables. Use ``input_transform`` for
    model-specific feature shaping and ``output_transform`` when a model returns
    vectors such as regime probabilities.
    """

    def __init__(
        self,
        members: list[EnsembleMember] | None = None,
        aggregation: str = "weighted_mean",
    ):
        if aggregation not in {"weighted_mean", "mean", "median"}:
            raise ValueError("aggregation must be weighted_mean, mean, or median")
        self.members = members or []
        self.aggregation = aggregation

    def add_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0,
        input_transform: InputTransform | None = None,
        output_transform: OutputTransform | None = None,
    ) -> None:
        self.members.append(
            EnsembleMember(
                name=name,
                model=model,
                weight=float(weight),
                input_transform=input_transform,
                output_transform=output_transform,
            )
        )

    def predict(self, x: np.ndarray) -> float:
        predictions = self._predict_weighted(x)
        if not predictions:
            raise ValueError("ensemble has no models")

        values = np.asarray([prediction for _, prediction in predictions], dtype=float)
        weights = np.asarray([weight for weight, _ in predictions], dtype=float)

        if self.aggregation == "median":
            return float(np.median(values))
        if self.aggregation == "mean":
            return float(np.mean(values))
        if weights.sum() <= 0:
            raise ValueError("at least one ensemble weight must be positive")
        return float(np.average(values, weights=weights))

    def predict_all(self, x: np.ndarray) -> dict[str, float]:
        return {
            member.name: prediction
            for member, prediction in self._predict_members(x)
        }

    def _predict_weighted(self, x: np.ndarray) -> list[tuple[float, float]]:
        return [
            (member.weight, prediction)
            for member, prediction in self._predict_members(x)
        ]

    def _predict_members(self, x: np.ndarray) -> list[tuple[EnsembleMember, float]]:
        features = np.asarray(x, dtype=float)
        predictions: list[tuple[EnsembleMember, float]] = []

        for member in self.members:
            if member.weight <= 0:
                continue

            model_input = (
                member.input_transform(features)
                if member.input_transform is not None
                else features
            )
            raw_prediction = self._predict_member(member.model, model_input)
            prediction = self._coerce_prediction(
                raw_prediction,
                member.output_transform,
                member.name,
            )
            predictions.append((member, prediction))

        return predictions

    def update(self, x: np.ndarray, y: float) -> None:
        for member in self.members:
            if hasattr(member.model, "update"):
                member.model.update(np.asarray(x, dtype=float), y)

    def _predict_member(self, model: Any, model_input: Any) -> Any:
        if hasattr(model, "predict"):
            return model.predict(model_input)
        if callable(model):
            return model(model_input)
        raise TypeError(f"model {model!r} does not provide predict or __call__")

    @staticmethod
    def _coerce_prediction(
        prediction: Any,
        output_transform: OutputTransform | None,
        name: str,
    ) -> float:
        if output_transform is not None:
            return float(output_transform(prediction))

        values = np.asarray(prediction, dtype=float).reshape(-1)
        if values.size != 1:
            raise ValueError(
                f"model {name!r} returned {values.size} values; provide "
                "output_transform to convert it to a scalar prediction"
            )
        return float(values[0])


class TorchModelAdapter(BaseModel):
    """Adapter for PyTorch models used as scalar predictors."""

    def __init__(
        self,
        model: Any,
        input_transform: InputTransform | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.input_transform = input_transform
        self.device = device

    def predict(self, x: np.ndarray) -> float:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ImportError("PyTorch is required for TorchModelAdapter") from exc

        values = self.input_transform(x) if self.input_transform else x
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(tensor)
        return float(prediction.detach().cpu().numpy().reshape(-1)[0])

    def update(self, x: np.ndarray, y: float) -> None:
        raise NotImplementedError("TorchModelAdapter does not train models online")


class KerasModelAdapter(BaseModel):
    """Adapter for Keras/TensorFlow models used as scalar predictors."""

    def __init__(
        self,
        model: Any,
        input_transform: InputTransform | None = None,
    ):
        self.model = model
        self.input_transform = input_transform

    def predict(self, x: np.ndarray) -> float:
        values = self.input_transform(x) if self.input_transform else x
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        prediction = self.model.predict(values, verbose=0)
        return float(np.asarray(prediction).reshape(-1)[0])

    def update(self, x: np.ndarray, y: float) -> None:
        raise NotImplementedError("KerasModelAdapter does not train models online")


class RegimeAwareEnsemble(ModelEnsemble):
    """Ensemble that appends GMM/HMM regime state before prediction."""

    def __init__(
        self,
        regime_model: BaseRegimeModel,
        transition_model: BaseTransitionModel | None = None,
        members: list[EnsembleMember] | None = None,
        aggregation: str = "weighted_mean",
    ):
        super().__init__(members=members, aggregation=aggregation)
        self.regime_model = regime_model
        self.transition_model = transition_model

    def predict(self, x: np.ndarray) -> float:
        return super().predict(self._augment_features(x))

    def predict_all(self, x: np.ndarray) -> dict[str, float]:
        return super().predict_all(self._augment_features(x))

    def update(self, x: np.ndarray, y: float) -> None:
        super().update(self._augment_features(x), y)

    def _augment_features(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float).reshape(-1)
        regime_probs = self.regime_model.update(features)
        regime_state = (
            self.transition_model.update(regime_probs)
            if self.transition_model is not None
            else regime_probs
        )
        return np.concatenate([features, np.asarray(regime_state, dtype=float).reshape(-1)])
