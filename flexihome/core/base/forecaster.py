"""Base forecasting plugin contracts for Residential VPP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Common forecast quality metrics returned by forecasting plugins."""

    mae: float
    rmse: float
    bias: float
    mape: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        extra = payload.pop("extra") or {}
        payload.update(extra)
        return {key: value for key, value in payload.items() if value is not None}


class BaseForecaster(ABC):
    """Abstract interface for pluggable forecasting models."""

    def __init__(self, name: str, version: str = "1.0.0", **metadata: Any) -> None:
        self.name = name
        self.version = version
        self.metadata = dict(metadata)
        self.is_fitted = False
        self.target: str | None = None
        self.predictors: List[str] = []
        self.lags = 0
        self.horizon_steps = 1
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.created_at = datetime.now(timezone.utc)

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseForecaster":
        """Fit the model with tabular training data."""

    @abstractmethod
    def predict(self, X_test: pd.DataFrame, horizon_steps: int = 1) -> Tuple[pd.Series, pd.Series | None]:
        """Predict future values and optionally return uncertainty estimates."""

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances when the model supports them."""

    def fit_from_frame(
        self,
        df: pd.DataFrame,
        target: str,
        predictors: Iterable[str],
        lags: int,
        horizon_steps: int,
        train_fraction: float = 0.8,
    ) -> pd.DataFrame:
        """Build lagged features from a time-series frame, fit, and return a comparison frame."""

        from flexihome.core.engine import build_feature_dataset

        predictors = list(predictors)
        X, y = build_feature_dataset(df, target, predictors, lags, horizon_steps)
        if X.empty:
            raise ValueError("Feature dataset is empty; increase input rows or reduce lags/horizon.")

        split = int(len(X) * train_fraction)
        split = min(max(split, 1), len(X) - 1) if len(X) > 1 else 1
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        self.target = target
        self.predictors = predictors
        self.lags = int(lags)
        self.horizon_steps = int(horizon_steps)
        self.feature_columns = list(X.columns)
        self.fit(X_train, y_train, X_val if not X_val.empty else None, y_val if not y_val.empty else None)

        if X_val.empty:
            predictions = pd.Series(dtype=float, name="prediction")
            uncertainty = None
            metrics = ForecastMetrics(mae=float("nan"), rmse=float("nan"), bias=float("nan"), mape=None)
        else:
            predictions, uncertainty = self.predict(X_val, horizon_steps=horizon_steps)
            predictions = pd.Series(predictions, index=X_val.index, name="prediction")
            metrics = self.evaluate(y_val, predictions)

        self.metrics = metrics.to_dict()
        comparison = pd.DataFrame({"actual": y_val, "prediction": predictions})
        if isinstance(uncertainty, pd.DataFrame) and not uncertainty.empty:
            for column in uncertainty.columns:
                comparison[f"prediction_{column}"] = uncertainty[column].reindex(comparison.index)
        return comparison

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> ForecastMetrics:
        """Calculate common forecast metrics."""

        actual = pd.Series(y_true).astype(float)
        predicted = pd.Series(y_pred, index=actual.index).astype(float)
        error = predicted - actual
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        bias = float(np.mean(error))
        nonzero = actual.replace(0, np.nan)
        percentage_errors = np.abs(error / nonzero).dropna()
        mape = float(percentage_errors.mean() * 100.0) if not percentage_errors.empty else None
        return ForecastMetrics(mae=mae, rmse=rmse, bias=bias, mape=mape)

    def get_metadata(self) -> Dict[str, Any]:
        """Return plugin metadata useful for audit logs and API diagnostics."""

        return {
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "target": self.target,
            "predictors": list(self.predictors),
            "lags": self.lags,
            "horizon_steps": self.horizon_steps,
            "metrics": dict(self.metrics),
            "created_at": self.created_at.isoformat(),
            **self.metadata,
        }

    def to_forecast_spec(self):
        """Return the legacy ForecastSpec wrapper when the plugin exposes a pickleable model."""

        from flexihome.core.engine import ForecastSpec

        model = getattr(self, "model", None)
        if model is None:
            raise ValueError(f"{self.__class__.__name__} does not expose a legacy model attribute.")
        if self.target is None:
            raise ValueError("Cannot convert an unfitted forecaster without a target.")
        return ForecastSpec(
            target=self.target,
            predictors=list(self.predictors),
            lags=int(self.lags),
            horizon_steps=int(self.horizon_steps),
            model=model,
            metrics=dict(self.metrics),
            residual_quantiles=dict(getattr(self, "residual_quantiles", {}) or {}),
        )
