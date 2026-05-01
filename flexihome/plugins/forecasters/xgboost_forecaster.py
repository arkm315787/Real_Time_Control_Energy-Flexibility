"""Production XGBoost forecasting plugin."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from xgboost import XGBRegressor

from flexihome.core.base.forecaster import BaseForecaster


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
        self.feature_columns = list(X_train.columns)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, horizon_steps: int = 1) -> tuple[pd.Series, pd.Series | None]:
        if self.model is None or not self.is_fitted:
            raise RuntimeError("XGBoostForecaster must be fitted before prediction.")
        predictions = pd.Series(self.model.predict(X_test), index=X_test.index, name="prediction")
        return predictions, None

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None or not self.is_fitted:
            return pd.DataFrame(columns=["feature", "importance"])
        feature_names = list(getattr(self.model, "feature_names_in_", self.feature_columns))
        return (
            pd.DataFrame({"feature": feature_names, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
