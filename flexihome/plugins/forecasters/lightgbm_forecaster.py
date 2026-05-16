"""LightGBM quantile forecasting plugin."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from flexihome.core.base.forecaster import BaseForecaster


FORECAST_QUANTILES = (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)


def _quantile_label(quantile: float) -> str:
    return f"p{int(round(float(quantile) * 100)):02d}"


class LightGBMQuantileForecaster(BaseForecaster):
    """Probabilistic forecaster trained as one gradient boosted model per quantile."""

    def __init__(
        self,
        name: str = "lightgbm_quantile",
        version: str = "1.0.0",
        seed: int = 42,
        model_params: Dict[str, Any] | None = None,
        **metadata: Any,
    ) -> None:
        super().__init__(name=name, version=version, seed=seed, **metadata)
        self.seed = int(seed)
        self.model_params = dict(model_params or {})
        self.quantiles = tuple(float(q) for q in FORECAST_QUANTILES)
        self.quantile_models: Dict[str, Any] = {}
        self.model: Any | None = None
        self.residual_quantiles: Dict[str, float] = {}
        self.backend = "lightgbm"
        self.import_error: str | None = None

    def _lightgbm_regressor_cls(self) -> Any | None:
        try:
            from lightgbm import LGBMRegressor

            self.import_error = None
            return LGBMRegressor
        except Exception as exc:
            self.import_error = str(exc)
            return None

    def _build_model(self, quantile: float, lgbm_regressor_cls: Any | None) -> Any:
        if lgbm_regressor_cls is not None:
            params = {
                "n_estimators": 260,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "quantile",
                "alpha": float(quantile),
                "random_state": self.seed,
                "verbosity": -1,
            }
            params.update(self.model_params)
            params["objective"] = "quantile"
            params["alpha"] = float(quantile)
            params.setdefault("random_state", self.seed)
            return lgbm_regressor_cls(**params)

        self.backend = "sklearn_gradient_boosting_quantile"
        params = {
            "n_estimators": 220,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "loss": "quantile",
            "alpha": float(quantile),
            "random_state": self.seed,
        }
        fallback_params = {
            key: value
            for key, value in self.model_params.items()
            if key in {"n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf", "min_samples_split"}
        }
        params.update(fallback_params)
        params["loss"] = "quantile"
        params["alpha"] = float(quantile)
        params.setdefault("random_state", self.seed)
        return GradientBoostingRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMQuantileForecaster":
        lgbm_regressor_cls = self._lightgbm_regressor_cls()
        self.backend = "lightgbm" if lgbm_regressor_cls is not None else "sklearn_gradient_boosting_quantile"
        self.quantile_models = {}
        for quantile in self.quantiles:
            label = _quantile_label(quantile)
            model = self._build_model(quantile, lgbm_regressor_cls)
            model.fit(X_train, pd.Series(y_train).astype(float))
            self.quantile_models[label] = model

        self.model = self.quantile_models.get("p50") or next(iter(self.quantile_models.values()))
        calibration_X = X_val if X_val is not None and not X_val.empty else X_train
        calibration_y = y_val if y_val is not None and not y_val.empty else y_train
        calibration_pred = pd.Series(self.model.predict(calibration_X), index=calibration_X.index)
        residuals = pd.Series(calibration_y, index=calibration_X.index).astype(float) - calibration_pred.astype(float)
        if residuals.empty:
            self.residual_quantiles = {_quantile_label(q): 0.0 for q in self.quantiles}
        else:
            self.residual_quantiles = {
                _quantile_label(q): float(np.nanquantile(residuals.to_numpy(dtype=float), q))
                for q in self.quantiles
            }
        self.feature_columns = list(X_train.columns)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, horizon_steps: int = 1) -> tuple[pd.Series, pd.DataFrame | None]:
        if not self.quantile_models or self.model is None or not self.is_fitted:
            raise RuntimeError("LightGBMQuantileForecaster must be fitted before prediction.")

        quantile_frame = pd.DataFrame(index=X_test.index)
        ordered_labels = [_quantile_label(q) for q in self.quantiles if _quantile_label(q) in self.quantile_models]
        for label in ordered_labels:
            quantile_frame[label] = self.quantile_models[label].predict(X_test)
        if ordered_labels:
            quantile_frame[ordered_labels] = np.maximum.accumulate(quantile_frame[ordered_labels].to_numpy(dtype=float), axis=1)

        predictions = pd.Series(quantile_frame.get("p50", self.model.predict(X_test)), index=X_test.index, name="prediction")
        quantile_frame["point"] = predictions
        return predictions, quantile_frame

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.quantile_models:
            return pd.DataFrame(columns=["feature", "importance"])
        importances = []
        for model in self.quantile_models.values():
            if hasattr(model, "feature_importances_"):
                importances.append(np.asarray(model.feature_importances_, dtype=float))
        if not importances:
            return pd.DataFrame(columns=["feature", "importance"])
        importance = np.mean(np.vstack(importances), axis=0)
        feature_names = list(getattr(self.model, "feature_names_in_", self.feature_columns))
        return (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_metadata(self) -> Dict[str, Any]:
        payload = super().get_metadata()
        payload["backend"] = self.backend
        if self.import_error:
            payload["lightgbm_import_error"] = self.import_error
        payload["quantiles"] = list(self.quantiles)
        return payload
