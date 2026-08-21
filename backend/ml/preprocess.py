#!/usr/bin/env python3
"""Data cleaning — the paper's *Data Pre-Processing* stage (section 4.2).

The paper asks for four things before any feature is built:

1. handling missing data,
2. removing duplicates,
3. converting categorical variables to numerical formats,
4. normalising / standardising.

(3) and (4) belong *inside* the fitted pipeline, not here, because they learn
parameters from the training split: one-hot categories and the scaler's
mean/scale are fitted in `ml.train_xgb` and travel inside the artifact. Doing
them here — on the full frame, before the split — is textbook data leakage.

So this module owns (1) and (2), plus target validation, and it returns a
`CleaningReport` that the trainer records in the model metadata. The imputation
values it learns are recorded too, so `app.services.model_service` can fill an
omitted optional field at request time with exactly the value training used,
instead of quietly defaulting to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from ml.features import TARGET, normalise_columns


@dataclass
class CleaningReport:
    """What cleaning actually did — recorded in the artifact, not just printed."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_missing_target: int = 0
    dropped_duplicates: int = 0
    imputed: dict[str, int] = field(default_factory=dict)
    impute_values: dict[str, Any] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"rows {self.rows_in:,} -> {self.rows_out:,}",
            f"dropped {self.dropped_missing_target} with no label",
            f"dropped {self.dropped_duplicates} duplicate rows",
        ]
        if self.imputed:
            parts.append("imputed " + ", ".join(f"{k}={v}" for k, v in self.imputed.items()))
        else:
            parts.append("no missing values to impute")
        return "  |  ".join(parts)


def clean(df: pd.DataFrame, *, target: str = TARGET,
          drop_duplicates: bool = True) -> tuple[pd.DataFrame, CleaningReport]:
    """Normalise headers, drop unlabelled and duplicate rows, impute the rest.

    Imputation follows the usual split by type: median for numeric columns
    (robust to the long right tail on follower counts, where a mean would be
    dragged by a handful of large accounts), and the empty string for text.
    """
    report = CleaningReport(rows_in=len(df))
    out = normalise_columns(df).copy()

    if target not in out.columns:
        raise ValueError(f"dataset has no target column {target!r}")

    # (1a) A row with no label cannot be trained on and cannot be imputed --
    # imputing the target is inventing ground truth.
    before = len(out)
    out = out[out[target].notna()]
    report.dropped_missing_target = before - len(out)

    # (2) Duplicate profiles bias the split: the same row can land in both train
    # and test, which inflates every metric.
    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        report.dropped_duplicates = before - len(out)

    # (1b) Impute what is left, recording both the count and the value used.
    for column in out.columns:
        if column == target:
            continue
        missing = int(out[column].isna().sum())
        if not missing:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            value: Any = float(out[column].median())
            if np.isnan(value):  # an entirely empty numeric column
                value = 0.0
        else:
            value = ""
        out[column] = out[column].fillna(value)
        report.imputed[column] = missing
        report.impute_values[column] = value

    # Record the median of every numeric column regardless of whether anything
    # was missing: the API needs a fill value for an optional field the caller
    # omits, and it must be the value the training distribution actually had.
    for column in out.columns:
        if column != target and pd.api.types.is_numeric_dtype(out[column]):
            report.impute_values.setdefault(column, float(out[column].median()))

    out[target] = out[target].astype(str)
    report.rows_out = len(out)
    counts = out[target].value_counts()
    report.classes = [str(c) for c in counts.index]
    report.class_counts = {str(k): int(v) for k, v in counts.items()}
    return out.reset_index(drop=True), report


def imbalance_ratio(class_counts: dict[str, int], positive: str) -> float:
    """negatives / positives — XGBoost's `scale_pos_weight` in its plainest form.

    1.0 means balanced. The paper notes fake profiles are the rarer class; this
    is what decides whether any imbalance handling is warranted at all, rather
    than applying it reflexively.
    """
    positives = class_counts.get(positive, 0)
    negatives = sum(v for k, v in class_counts.items() if k != positive)
    if positives == 0:
        return float("inf")
    return negatives / positives
