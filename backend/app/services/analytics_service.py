"""Analytics, computed once at startup.

`website/app.py:analytics` re-read a 15,000-row CSV with pandas, recomputed a
histogram and re-derived feature names *on every page load*, inside the request
thread. At 100 concurrent users that is 100 concurrent full CSV parses. All of it
is static between deploys, so it is computed once here and served from memory.

If a dataset is not configured the service still works -- it falls back to the
counts recorded in the model metadata at training time and simply omits the
histogram, instead of silently rendering the hardcoded placeholder numbers the
original template fell back to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.schemas.analytics import (
    AnalyticsData,
    ClassDistribution,
    FeatureImportance,
    HistogramBin,
    ModelMetrics,
)

logger = get_logger(__name__)

_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
_BIN_LABELS = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%", "5-6%", "6-7%", "7-8%",
               "8-9%", "9-10%", "10-15%", "15-20%+"]


class AnalyticsService:
    def __init__(self, dataset_path: Path | None) -> None:
        self._dataset_path = dataset_path
        self._cached: AnalyticsData | None = None

    def build(self, model_meta: dict[str, Any]) -> None:
        """Compute the snapshot. Called once from the lifespan handler."""
        metrics = ModelMetrics(**model_meta["metrics"])
        importances = [
            FeatureImportance(**fi) for fi in model_meta.get("top_feature_importances", [])
        ]

        distribution: list[ClassDistribution] = []
        histogram: list[HistogramBin] = []
        samples = int(model_meta.get("training_samples", 0))

        # is_file(), not exists(): a directory passes exists() and then explodes
        # inside read_csv. See the DATASET_PATH validator in app.core.config.
        if self._dataset_path and self._dataset_path.is_file():
            try:
                df = pd.read_csv(
                    self._dataset_path,
                    usecols=["Account Type", "Engagement Rate (%)"],
                )
                samples = len(df)
                counts = df["Account Type"].value_counts()
                total = int(counts.sum()) or 1
                distribution = [
                    ClassDistribution(
                        label=str(label),
                        count=int(count),
                        percentage=round(count / total * 100, 2),
                    )
                    for label, count in counts.items()
                ]
                hist, _ = np.histogram(df["Engagement Rate (%)"].to_numpy(), bins=_BINS)
                histogram = [
                    HistogramBin(label=lbl, count=int(c))
                    for lbl, c in zip(_BIN_LABELS, hist)
                ]
            except Exception:
                # Analytics is a nice-to-have; never let it stop the service booting.
                logger.exception("analytics_dataset_read_failed",
                                 extra={"path": str(self._dataset_path)})
        else:
            logger.warning("analytics_dataset_missing",
                           extra={"path": str(self._dataset_path)})

        self._cached = AnalyticsData(
            training_samples=samples,
            feature_count=int(model_meta.get("feature_count", 0)),
            class_distribution=distribution,
            feature_importances=importances,
            engagement_histogram=histogram,
            metrics=metrics,
            model_version=str(model_meta.get("model_version", "unknown")),
            trained_at=str(model_meta.get("trained_at", "")),
        )
        logger.info("analytics_ready", extra={"samples": samples})

    @property
    def snapshot(self) -> AnalyticsData | None:
        return self._cached
