from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    target_column: str
    horizons_hours: tuple[int, ...]
    history_hours: int
    train_end: str
    validation_end: str
    test_end: str | None
    random_seed: int
    random_forest: dict[str, int | None]
    multivariate_columns: tuple[str, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        config = cls(
            target_column=raw["target_column"],
            horizons_hours=tuple(raw["horizons_hours"]),
            history_hours=int(raw["history_hours"]),
            train_end=raw["train_end"],
            validation_end=raw["validation_end"],
            test_end=raw.get("test_end"),
            random_seed=int(raw["random_seed"]),
            random_forest=raw["random_forest"],
            multivariate_columns=tuple(raw["multivariate_columns"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.horizons_hours or any(value <= 0 for value in self.horizons_hours):
            raise ValueError("Forecast horizons must be positive integers.")
        if self.history_hours <= 0:
            raise ValueError("history_hours must be positive.")
        if self.target_column not in self.multivariate_columns:
            raise ValueError(
                "The target column must be present in multivariate_columns."
            )
