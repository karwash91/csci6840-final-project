from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
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
    validation_history_hours: tuple[int, ...] = ()
    random_forest_candidates: tuple[dict[str, int | None], ...] = ()
    selected_models: dict[str, dict[str, object]] = field(default_factory=dict)

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
            validation_history_hours=tuple(raw.get("validation_history_hours", ())),
            random_forest_candidates=tuple(raw.get("random_forest_candidates", ())),
            selected_models=raw.get("selected_models", {}),
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
        if any(value <= 0 for value in self.history_candidates):
            raise ValueError("Validation history lengths must be positive integers.")

    @property
    def history_candidates(self) -> tuple[int, ...]:
        return self.validation_history_hours or (self.history_hours,)

    @property
    def forest_candidates(self) -> tuple[dict[str, int | None], ...]:
        return self.random_forest_candidates or (self.random_forest,)

    @property
    def is_frozen(self) -> bool:
        expected = {
            self.selection_key(horizon, model, input_name)
            for horizon in self.horizons_hours
            for model in ("linear_regression", "random_forest")
            for input_name in ("temperature_only", "multivariate")
        }
        return expected.issubset(self.selected_models)

    def selection_key(self, horizon: int, model: str, input_name: str) -> str:
        return f"h{horizon:02d}_{model}_{input_name}"

    def selection_for(
        self, horizon: int, model: str, input_name: str
    ) -> tuple[int, dict[str, int | None]]:
        selected = self.selected_models.get(
            self.selection_key(horizon, model, input_name), {}
        )
        history = int(selected.get("history_hours", self.history_hours))
        forest = selected.get("random_forest", self.random_forest)
        return history, dict(forest)

    def with_selections(
        self, selections: dict[str, dict[str, object]]
    ) -> "ExperimentConfig":
        return replace(self, selected_models=selections)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
