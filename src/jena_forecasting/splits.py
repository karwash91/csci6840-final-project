from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import SupervisedFrame


@dataclass(frozen=True)
class Partition:
    features: pd.DataFrame
    target: pd.Series
    target_time: pd.Series


def chronological_partitions(
    supervised: SupervisedFrame,
    train_end: str,
    validation_end: str,
    test_end: str | None,
) -> dict[str, Partition]:
    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)
    test_boundary = pd.Timestamp(test_end) if test_end else None
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be earlier than validation_end.")
    if test_boundary is not None and validation_boundary >= test_boundary:
        raise ValueError("validation_end must be earlier than test_end.")
    target_time = supervised.target_time
    masks = {
        "train": target_time <= train_boundary,
        "validation": (target_time > train_boundary)
        & (target_time <= validation_boundary),
        "test": target_time > validation_boundary,
    }
    if test_boundary is not None:
        masks["test"] &= target_time <= test_boundary
    return {
        name: Partition(
            features=supervised.features.loc[mask],
            target=supervised.target.loc[mask],
            target_time=target_time.loc[mask],
        )
        for name, mask in masks.items()
    }
