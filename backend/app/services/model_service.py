"""Model loading and inference.

Design notes, each answering a specific defect in the original Flask app:

*   **Load once, at startup.** `website/app.py` had a module-level `load_model()`
    plus a lazy re-load inside the `/analytics` view guarded by `global` -- under
    a threaded server two requests could re-load a 72 MB pickle concurrently.
    Here the artifact is loaded once in the lifespan handler and the instance is
    immutable afterwards.
*   **Fail fast on version mismatch.** The artifact records the sklearn version
    that produced it. If the running version differs, we refuse to serve rather
    than returning a 500 on every prediction.
*   **Bounded concurrency.** `predict` is CPU-bound. It runs in a worker thread
    behind a semaphore so a burst cannot spawn unbounded threads or queue
    unbounded memory.
*   **No mutation of caller input.** The old `make_prediction` mutated the dict it
    was handed (`input_data['Bio Text'] = bio_text_encoded`), which is a landmine
    for retries.
*   **One feature implementation, not two.** For the paper's XGBoost artifact the
    request is turned into features by `ml.features.build_feature_frame` — the
    exact function `ml.train_xgb` called during fitting. Nothing here recomputes
    a feature by hand, which is how training-serving skew starts, and the built
    column list is asserted against the list recorded at training time so a
    feature added to `ml.features` without retraining fails at startup rather
    than silently shifting every column.

Two artifact styles are supported, decided by the metadata:

* **engineered** (`feature_columns` present) — the paper pipeline from
  `ml.train_xgb`. Raw profile fields in, `build_feature_frame` applied here.
* **legacy** (no `feature_columns`) — the RandomForest from `ml.train`, whose
  own pipeline consumes the CSV columns directly. Still loadable, so the older
  artifact and anything trained against it keep working.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import anyio
import joblib
import pandas as pd
import sklearn

from app.core.errors import InferenceError, InferenceTimeoutError, ModelNotReadyError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Wire field -> the column name the trained pipeline expects. Single source of
# truth for the mapping; the schema layer never needs to know the CSV headers.
FEATURE_COLUMNS: dict[str, str] = {
    "followers": "Followers",
    "following": "Following",
    "posts": "Posts",
    "engagement_rate": "Engagement Rate (%)",
    "avg_likes_per_post": "Avg Likes per Post",
    "avg_comments_per_post": "Avg Comments per Post",
    "verified": "Verified",
    "account_age_years": "Account Age (Years)",
    "bio_text": "Bio Text",
}
COLUMN_ORDER = list(FEATURE_COLUMNS.values())


class ModelService:
    """Holds the fitted pipeline and serves predictions."""

    def __init__(self, model_path: Path, metadata_path: Path, *,
                 max_concurrency: int = 8, timeout_seconds: float = 5.0) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._timeout = timeout_seconds
        self._limiter = anyio.CapacityLimiter(max_concurrency)
        self._model: Any = None
        self._meta: dict[str, Any] = {}
        # Set during load() from the metadata; see the module docstring.
        self._style = "legacy"
        self._feature_columns: list[str] = []
        self._impute_values: dict[str, Any] = {}
        self._labels: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if not self._model_path.exists():
            raise ModelNotReadyError(
                f"Model artifact not found at {self._model_path}. Train it first:\n"
                "  python -m ml.generate_dataset --rows 50000 --out ../data/paper_signal.csv\n"
                f"  python -m ml.train_xgb --dataset ../data/paper_signal.csv "
                f"--out {self._model_path.parent}\n"
                "(For the older RandomForest artifact, use `python -m ml.train` and "
                "point ARTIFACTS_DIR at its output directory.)"
            )
        if not self._metadata_path.exists():
            raise ModelNotReadyError(f"Model metadata not found at {self._metadata_path}.")

        self._meta = json.loads(self._metadata_path.read_text(encoding="utf-8"))

        trained_with = self._meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            raise ModelNotReadyError(
                f"Artifact was trained with scikit-learn {trained_with} but this "
                f"process is running {sklearn.__version__}. Loading across minor "
                "versions is not supported; retrain or pin the matching version."
            )

        # Same guarantee for the booster: an XGBoost artifact unpickled on a
        # different major version can load and then score differently.
        xgb_trained_with = self._meta.get("xgboost_version")
        if xgb_trained_with:
            try:
                import xgboost
            except ImportError as exc:  # pragma: no cover - deployment error
                raise ModelNotReadyError(
                    "Artifact is an XGBoost model but xgboost is not installed."
                ) from exc
            if xgb_trained_with.split(".")[0] != xgboost.__version__.split(".")[0]:
                raise ModelNotReadyError(
                    f"Artifact was trained with xgboost {xgb_trained_with} but this "
                    f"process is running {xgboost.__version__}."
                )

        # Which feature path this artifact needs.
        self._feature_columns = list(self._meta.get("feature_columns") or [])
        self._style = "engineered" if self._feature_columns else "legacy"
        self._impute_values = dict(
            (self._meta.get("cleaning") or {}).get("impute_values") or {}
        )
        self._labels = [str(c) for c in (self._meta.get("classes") or [])]

        started = time.perf_counter()
        self._model = joblib.load(self._model_path)
        elapsed = (time.perf_counter() - started) * 1000

        logger.info(
            "model_loaded",
            extra={
                "model_version": self.version,
                "algorithm": self._meta.get("algorithm", "unknown"),
                "feature_style": self._style,
                "feature_count": len(self._feature_columns) or None,
                "load_ms": round(elapsed, 1),
                "size_mb": round(self._model_path.stat().st_size / 1e6, 2),
                "sklearn_version": sklearn.__version__,
            },
        )
        for w in self._meta.get("warnings", []):
            logger.warning("model_quality_warning", extra={"detail": w})

    def unload(self) -> None:
        self._model = None

    # -- introspection -----------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return str(self._meta.get("model_version", "unknown"))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta

    @property
    def classes(self) -> list[str]:
        # The paper artifact is fitted on 0/1, so `classes_` is [0, 1]; the human
        # labels live in the metadata, written in the same order.
        if self._labels:
            return list(self._labels)
        return [str(c) for c in getattr(self._model, "classes_", [])]

    @property
    def feature_style(self) -> str:
        return self._style

    # -- inference ---------------------------------------------------------

    @staticmethod
    def _to_legacy_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
        """Build one DataFrame for the whole batch, in the column order the
        pipeline was fitted on. Booleans become ints because the numeric branch
        was fitted on 0/1."""
        rows = []
        for item in items:
            row = {FEATURE_COLUMNS[k]: v for k, v in item.items() if k in FEATURE_COLUMNS}
            row["Verified"] = int(bool(row.get("Verified", 0)))
            row["Bio Text"] = str(row.get("Bio Text", ""))
            rows.append(row)
        return pd.DataFrame(rows, columns=COLUMN_ORDER)

    def _to_engineered_frame(
        self, items: list[dict[str, Any]]
    ) -> tuple[pd.DataFrame, list[list[str]]]:
        """Raw payload -> the feature frame the paper pipeline was fitted on.

        The wire field names are already the canonical raw names
        `ml.features` expects, so the payload goes in unchanged apart from two
        things:

        * booleans become 0/1, because training saw integers;
        * a field the caller omitted is filled with the value the *training*
          data would have imputed (`ml.preprocess.clean` recorded it), not a
          silent zero. Which fields that happened to is returned, so the
          response can tell the caller their answer was scored with defaults.
        """
        from ml import features as ml_features

        rows, imputed_per_row = [], []
        for item in items:
            row: dict[str, Any] = {}
            imputed: list[str] = []
            for key, value in item.items():
                if value is None:
                    continue
                row[key] = int(value) if isinstance(value, bool) else value
            for column in ml_features.COLUMN_ALIASES:
                if column in row:
                    continue
                if column in self._impute_values:
                    row[column] = self._impute_values[column]
                    imputed.append(column)
                elif column in ("full_name", "bio_text", "language"):
                    row[column] = ""
                    imputed.append(column)
            rows.append(row)
            imputed_per_row.append(imputed)

        frame = ml_features.build_feature_frame(pd.DataFrame(rows))

        missing = [c for c in self._feature_columns if c not in frame.columns]
        if missing:
            # A feature the model was fitted on that this request cannot produce.
            # Refusing is the only safe answer: silently filling it would score
            # the profile against a column the model has never seen empty.
            raise InferenceError(
                f"Request cannot produce {len(missing)} feature(s) the model was "
                "trained on. The artifact and ml/features.py are out of step; retrain."
            )
        return frame[self._feature_columns], imputed_per_row

    def _predict_sync(self, frame: pd.DataFrame) -> tuple[list[str], list[dict[str, float]]]:
        proba = self._model.predict_proba(frame)
        classes = self.classes or [str(c) for c in self._model.classes_]
        labels = [classes[int(i)] for i in proba.argmax(axis=1)]
        dists = [{c: round(float(p), 6) for c, p in zip(classes, row)} for row in proba]
        return labels, dists

    async def predict(
        self, items: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, float]], float, list[list[str]]]:
        """Score a batch.

        Returns (labels, per-class probabilities, latency_ms, imputed field names
        per item). The last element is empty for every item on the legacy
        artifact, which has no optional fields.
        """
        if not self.is_ready:
            raise ModelNotReadyError()

        if self._style == "engineered":
            frame, imputed = self._to_engineered_frame(items)
        else:
            frame, imputed = self._to_legacy_frame(items), [[] for _ in items]
        started = time.perf_counter()
        try:
            with anyio.fail_after(self._timeout):
                labels, dists = await anyio.to_thread.run_sync(
                    self._predict_sync, frame, limiter=self._limiter
                )
        except TimeoutError as exc:
            raise InferenceTimeoutError(
                f"Prediction exceeded {self._timeout}s."
            ) from exc
        except Exception as exc:
            # Log the real cause; the client gets a stable code, never the trace.
            logger.exception("inference_failed", extra={"exc_type": type(exc).__name__})
            raise InferenceError() from exc

        return labels, dists, round((time.perf_counter() - started) * 1000, 2), imputed
