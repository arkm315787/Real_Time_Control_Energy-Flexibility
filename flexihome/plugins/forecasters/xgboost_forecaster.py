"""Production XGBoost forecasting plugin."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from flexihome.core.base.forecaster import BaseForecaster


FORECAST_QUANTILES = (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)


def _quantile_label(quantile: float) -> str:
    return f"p{int(round(float(quantile) * 100)):02d}"


class XGBoostForecaster(BaseForecaster):
    """XGBoost implementation of the FlexiHome forecaster contract."""

    def __init__(
        self,
        name: str = "xgboost_default",
        version: str = "1.0.0",
        seed: int = 42,
        model_params: Dict[str, Any] | None = None,
        **metadata: Any,
    ) -> None:
        super().__init__(name=name, version=version, seed=seed, **metadata)
        self.seed = int(seed)
        self.model_params = dict(model_params or {})
        self.model: XGBRegressor | None = None
        self.residual_quantiles: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostForecaster":
        params = {
            "n_estimators": 220,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": self.seed,
        }
        params.update(self.model_params)
        self.model = XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        calibration_X = X_val if X_val is not None and not X_val.empty else X_train
        calibration_y = y_val if y_val is not None and not y_val.empty else y_train
        calibration_pred = pd.Series(self.model.predict(calibration_X), index=calibration_X.index)
        residuals = pd.Series(calibration_y, index=calibration_X.index).astype(float) - calibration_pred.astype(float)
        if residuals.empty:
            self.residual_quantiles = {_quantile_label(q): 0.0 for q in FORECAST_QUANTILES}
        else:
            self.residual_quantiles = {
                _quantile_label(q): float(np.nanquantile(residuals.to_numpy(dtype=float), q))
                for q in FORECAST_QUANTILES
            }
        self.feature_columns = list(X_train.columns)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, horizon_steps: int = 1) -> tuple[pd.Series, pd.Series | None]:
        if self.model is None or not self.is_fitted:
            raise RuntimeError("XGBoostForecaster must be fitted before prediction.")
        predictions = pd.Series(self.model.predict(X_test), index=X_test.index, name="prediction")
        quantiles = pd.DataFrame(index=X_test.index)
        for label, residual in self.residual_quantiles.items():
            quantiles[label] = predictions + float(residual)
        quantiles["point"] = predictions
        return predictions, quantiles

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or not self.is_fitted:
            return pd.DataFrame(columns=["feature", "importance"])
        feature_names = list(getattr(self.model, "feature_names_in_", self.feature_columns))
        return (
            pd.DataFrame({"feature": feature_names, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
