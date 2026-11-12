from __future__ import annotations

from typing import Any


def make_model(name: str, random_seed: int, random_forest: dict[str, Any]) -> Any:
    if name == "linear_regression":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        settings = dict(random_forest)
        settings["random_state"] = random_seed
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(**settings)),
            ]
        )
    raise ValueError(f"Unknown model: {name}")
