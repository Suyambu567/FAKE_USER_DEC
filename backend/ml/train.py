#!/usr/bin/env python3
"""Train and export the fake-profile classifier.

This replaces `FAKE_PROFILE_TRAIN_CODE/train.py`, `train_improved.py` and
`debug.py`, which were three near-identical copies of the same script.

Fixes carried over from the audit:

*   **No `bio_encoder.pkl`.** The old app label-encoded bio text to an integer and
    fed that to a `TfidfVectorizer`, which crashes with
    `'int' object has no attribute 'lower'`. Bio text stays a raw string and is
    vectorised inside the pipeline, so the artifact is self-contained.
*   **Bounded tree size.** The old forest was 200 unbounded-ish trees / ~300k nodes
    / 72 MB, memorising noise (train 0.96 vs test 0.51). Depth and leaf size are
    now constrained, which cuts the artifact by an order of magnitude and makes
    per-request memory predictable.
*   **Honest evaluation.** Reports accuracy *next to* the majority-class baseline
    and cross-validated accuracy, and writes a machine-readable warning into the
    metadata when the model shows no lift.
*   **Reproducible metadata.** Records the sklearn version that produced the
    artifact so the API can refuse to load a mismatched pickle instead of failing
    at the first prediction.

Usage:
    python -m ml.train --dataset ../website/dataset.csv --out artifacts/
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Account Type"
TEXT_COL = "Bio Text"

# Wire name -> dataset column. The API speaks snake_case; the CSV does not.
FEATURE_MAP: dict[str, str] = {
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
NUMERIC_COLS = [c for k, c in FEATURE_MAP.items() if k != "bio_text"]

RANDOM_STATE = 42


def build_pipeline(n_estimators: int, max_depth: int, min_samples_leaf: int) -> Pipeline:
    """Numeric scaling + TF-IDF over raw bio text, then a bounded random forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "txt",
                TfidfVectorizer(
                    max_features=200,
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=2,
                ),
                TEXT_COL,  # a bare string selects a 1-D Series, which TF-IDF requires
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def extract_feature_names(pipeline: Pipeline) -> list[str]:
    """Real post-transform feature names, with a safe fallback.

    The old analytics view hand-rolled this by walking `transformers_` and silently
    fell back to nine hardcoded placeholder importances when it failed -- which it
    always did, because the transformed space has 200+ TF-IDF columns, not 9.
    """
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        return NUMERIC_COLS + [f"tfidf_{i}" for i in range(200)]


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
             X: pd.DataFrame, y: pd.Series, positive: str) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    pos_idx = list(pipeline.classes_).index(positive)

    baseline = float(y_test.value_counts(normalize=True).max())
    accuracy = float(accuracy_score(y_test, y_pred))

    cv = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", n_jobs=-1,
    )

    return {
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline, 4),
        "lift_over_baseline": round(accuracy - baseline, 4),
        "precision": round(float(precision_score(y_test, y_pred, pos_label=positive, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, pos_label=positive, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, pos_label=positive, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score((y_test == positive).astype(int), proba[:, pos_idx])), 4),
        "cv_accuracy_mean": round(float(cv.mean()), 4),
        "cv_accuracy_std": round(float(cv.std()), 4),
    }


def audit_dataset(df: pd.DataFrame) -> list[str]:
    """Flag data problems that would otherwise be invisible behind a trained model."""
    warnings: list[str] = []

    n_bios = df[TEXT_COL].nunique()
    if n_bios < 50:
        warnings.append(
            f"Bio Text has only {n_bios} distinct values across {len(df)} rows; "
            "it carries almost no signal and the TF-IDF features are near-constant."
        )

    # Standardised mean difference per numeric feature. Under a real signal at
    # least one feature separates the classes; under random labels none do.
    max_smd, worst = 0.0, ""
    for col in NUMERIC_COLS:
        groups = df.groupby(TARGET)[col]
        if groups.ngroups != 2:
            continue
        means = groups.mean()
        sd = df[col].std() or 1.0
        smd = abs(means.iloc[0] - means.iloc[1]) / sd
        if smd > max_smd:
            max_smd, worst = smd, col
    if max_smd < 0.1:
        warnings.append(
            f"No numeric feature separates the classes (largest standardised mean "
            f"difference is {max_smd:.4f} on '{worst}'). The labels appear to be "
            "statistically independent of the features -- consistent with "
            "data/app.py generating 'Account Type' via np.random.choice. "
            "No model can beat the majority-class baseline on this data."
        )
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--n-estimators", type=int, default=120)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-samples-leaf", type=int, default=20)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--note", action="append", default=[],
                    help="Extra caveat to record in model_meta.json's warnings, so it "
                         "reaches every consumer of the model. Repeatable.")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    operator_notes = list(args.note)
    df = pd.read_csv(args.dataset)
    missing = [c for c in (*NUMERIC_COLS, TEXT_COL, TARGET) if c not in df.columns]
    if missing:
        print(f"error: dataset is missing columns: {missing}", file=sys.stderr)
        return 1

    before = len(df)
    df = df.dropna(subset=[*NUMERIC_COLS, TEXT_COL, TARGET])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    if len(df) < before:
        print(f"dropped {before - len(df)} rows with nulls")

    print(f"dataset: {args.dataset}  rows={len(df)}  classes={df[TARGET].unique().tolist()}")
    warnings = operator_notes + audit_dataset(df)
    for w in warnings:
        print(f"  ! WARNING: {w}")

    X = df[[*NUMERIC_COLS, TEXT_COL]]
    y = df[TARGET].astype(str)
    positive = "Fake" if "Fake" in set(y) else sorted(set(y))[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(args.n_estimators, args.max_depth, args.min_samples_leaf)
    print("training...")
    pipeline.fit(X_train, y_train)

    print("evaluating...")
    metrics = evaluate(pipeline, X_test, y_test, X, y, positive)

    if metrics["lift_over_baseline"] <= 0.01:
        warnings.append(
            f"Model accuracy ({metrics['accuracy']}) does not meaningfully exceed the "
            f"majority-class baseline ({metrics['baseline_accuracy']}). Predictions "
            "are not trustworthy and must not be used to action real accounts."
        )

    names = extract_feature_names(pipeline)
    importances = pipeline.named_steps["classifier"].feature_importances_
    top = sorted(
        ({"feature": n, "importance": round(float(v), 6)} for n, v in zip(names, importances)),
        key=lambda d: d["importance"], reverse=True,
    )[:20]

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    meta = {
        "model_version": version,
        "algorithm": "RandomForestClassifier + TF-IDF(bio) + StandardScaler(numeric)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset": str(args.dataset),
        "training_samples": int(len(df)),
        "feature_count": int(len(names)),
        "classes": [str(c) for c in pipeline.classes_],
        "positive_class": positive,
        "feature_map": FEATURE_MAP,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        },
        "metrics": metrics,
        "top_feature_importances": top,
        "warnings": warnings,
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_mb = model_path.stat().st_size / 1e6
    nodes = sum(t.tree_.node_count for t in pipeline.named_steps["classifier"].estimators_)
    print(f"\nsaved {model_path} ({size_mb:.2f} MB, {nodes:,} tree nodes)")
    print(f"saved {args.out / 'model_meta.json'}")
    print("\nmetrics:")
    for k, v in metrics.items():
        print(f"  {k:22s} {v}")
    if warnings:
        print(f"\n{len(warnings)} warning(s) recorded in model_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
