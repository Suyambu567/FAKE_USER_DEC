#!/usr/bin/env python3
"""Train the XGBoost fake-profile classifier — the paper's main pipeline.

Paper: *Fake Profile Detection Using XGBoost Algorithm* (Venkadesh et al.,
ICRDICCT'25), sections 4.2 to 4.5.

    Data Collection -> Pre-Processing -> Feature Engineering -> XGBoost
    -> Evaluation -> Model Saving -> (ml.visualize, app/ REST API)

This sits alongside `ml.train`, which trains the RandomForest that the static
`webapp/` ships. That model still works and is still exported to JSON for the
browser; nothing here replaces it. This is the paper's algorithm, with the
paper's feature set, saved as its own artifact.

Stage-by-stage, and where each lives:

* **Pre-processing** — `ml.preprocess.clean`: missing values imputed, duplicate
  rows removed, unlabelled rows dropped.
* **Feature engineering** — `ml.features.build_feature_frame`: Table 1 profile
  features, behavioural/activity features, textual features. The *same* function
  runs at request time, so there is no training-serving skew.
* **Encoding and normalisation** — inside the fitted pipeline (one-hot for the
  categorical branch, TF-IDF for the text branch, StandardScaler for numerics),
  because all three learn parameters and fitting them before the split leaks the
  test set. Note that a tree ensemble is scale-invariant, so the scaler changes
  nothing mathematically here; it is in the pipeline because the paper's
  preprocessing stage calls for standardisation and it costs one pass.
* **Class imbalance** — measured first, then handled only if the data is
  actually imbalanced, and only ever on the training split.
* **Evaluation** — `ml.evaluation`, including the confusion matrix.

Usage:
    python -m ml.train_xgb --dataset ../data/paper_signal.csv --out artifacts/xgb
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import evaluation, features
from ml.preprocess import clean, imbalance_ratio

RANDOM_STATE = 42
# The strategy is chosen automatically unless the operator overrides it, and the
# choice lands in the metadata either way.
IMBALANCE_CHOICES = ("auto", "none", "class-weight", "oversample")


def build_pipeline(numeric: list[str], text: str | None, categorical: str | None,
                   *, params: dict, scale_pos_weight: float | None) -> Pipeline:
    """ColumnTransformer over the three feature groups, then XGBoost."""
    from xgboost import XGBClassifier

    transformers: list[tuple] = [("num", StandardScaler(), numeric)]
    if text:
        transformers.append((
            "txt",
            TfidfVectorizer(max_features=200, stop_words="english", lowercase=True,
                            strip_accents="unicode", min_df=2),
            text,  # a bare string selects a 1-D Series, which TF-IDF requires
        ))
    if categorical:
        transformers.append((
            "cat",
            # handle_unknown="ignore": a language absent from training must not
            # change the column layout at request time.
            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
            [categorical],
        ))

    # verbose_feature_names_out=True prefixes each branch (num__/txt__/cat__).
    # It is required, not cosmetic: the numeric column `followers` and the
    # TF-IDF token `followers` from bio text collide otherwise, and sklearn
    # refuses to name the output at all.
    preprocessor = ColumnTransformer(transformers, remainder="drop",
                                     verbose_feature_names_out=True)

    classifier = XGBClassifier(
        objective="binary:logistic",   # paper eq. (2): binary log loss
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    if scale_pos_weight is not None:
        classifier.set_params(scale_pos_weight=scale_pos_weight)

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def oversample(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Random oversampling of the minority class — training split only.

    The paper names "oversampling the minority class" as one of its two options.
    Called after `train_test_split`, never before: duplicating rows first would
    put copies of the same profile on both sides of the split and inflate every
    metric.
    """
    counts = y.value_counts()
    target = int(counts.max())
    rng = np.random.default_rng(seed)
    parts_X, parts_y = [X], [y]
    for label, n in counts.items():
        deficit = target - int(n)
        if deficit <= 0:
            continue
        pool = np.flatnonzero((y == label).to_numpy())
        picks = rng.choice(pool, size=deficit, replace=True)
        parts_X.append(X.iloc[picks])
        parts_y.append(y.iloc[picks])
    X_out = pd.concat(parts_X, ignore_index=True)
    y_out = pd.concat(parts_y, ignore_index=True)
    order = rng.permutation(len(X_out))
    return X_out.iloc[order].reset_index(drop=True), y_out.iloc[order].reset_index(drop=True)


def choose_strategy(requested: str, ratio: float) -> tuple[str, str]:
    """(strategy, why). `auto` only acts when the data is genuinely skewed."""
    if requested != "auto":
        return requested, f"requested explicitly (--imbalance {requested})"
    if 0.8 <= ratio <= 1.25:
        return "none", (f"negatives/positives = {ratio:.2f}; the classes are close "
                        "enough to balanced that reweighting would only add variance")
    return "class-weight", (f"negatives/positives = {ratio:.2f}; XGBoost's "
                            "scale_pos_weight is set to that ratio")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/xgb"))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--subsample", type=float, default=0.85)
    ap.add_argument("--colsample-bytree", type=float, default=0.85)
    ap.add_argument("--reg-lambda", type=float, default=1.5,
                    help="L2 term, lambda in the paper's regularisation eq. (3).")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="Minimum split loss, gamma in the paper's eq. (3).")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5, help="0 disables cross-validation.")
    ap.add_argument("--imbalance", choices=IMBALANCE_CHOICES, default="auto")
    ap.add_argument("--no-duplicate-removal", action="store_true",
                    help="Keep duplicate rows (for measuring what removing them changes).")
    ap.add_argument("--note", action="append", default=[],
                    help="Extra caveat recorded in the metadata warnings. Repeatable.")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    # ---- stage 1: data collection ---------------------------------------
    print(f"[1/6] loading {args.dataset}")
    raw = pd.read_csv(args.dataset)
    print(f"      {len(raw):,} rows x {len(raw.columns)} columns")

    # ---- stage 2: pre-processing ----------------------------------------
    print("[2/6] pre-processing (missing values, duplicates)")
    df, report = clean(raw, drop_duplicates=not args.no_duplicate_removal)
    print(f"      {report.summary()}")
    if len(report.classes) != 2:
        print(f"error: expected 2 classes, found {report.classes}", file=sys.stderr)
        return 1

    positive = features.POSITIVE_CLASS if features.POSITIVE_CLASS in report.classes else report.classes[0]
    negative = next(c for c in report.classes if c != positive)
    # Explicit mapping rather than LabelEncoder's alphabetical order: XGBoost's
    # scale_pos_weight is defined against class 1, which must be the fake class.
    label_to_int = {negative: 0, positive: 1}
    int_to_label = {v: k for k, v in label_to_int.items()}

    # ---- stage 3: feature engineering ------------------------------------
    print("[3/6] feature engineering")
    X = features.build_feature_frame(df)
    y = df[features.TARGET].astype(str)
    numeric, text, categorical = features.split_columns(X)
    avail = features.availability(set(features.normalise_columns(df).columns))
    print(f"      {len(X.columns)} features: {len(numeric)} numeric"
          f"{', 1 text (TF-IDF)' if text else ''}"
          f"{', 1 categorical (one-hot)' if categorical else ''}")
    print(f"      paper Table 1 attributes available:   {', '.join(avail['available'])}")
    if avail["unavailable"]:
        print(f"      paper Table 1 attributes UNAVAILABLE: {', '.join(avail['unavailable'])}"
              "  (no source column in this dataset — not fabricated)")

    # ---- stage 4: split, then imbalance handling -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    ratio = imbalance_ratio(report.class_counts, positive)
    strategy, why = choose_strategy(args.imbalance, ratio)
    print(f"[4/6] class imbalance: {report.class_counts}")
    print(f"      strategy '{strategy}' — {why}")

    scale_pos_weight = None
    if strategy == "class-weight":
        train_counts = y_train.value_counts().to_dict()
        scale_pos_weight = imbalance_ratio({str(k): int(v) for k, v in train_counts.items()}, positive)
    elif strategy == "oversample":
        before = len(X_train)
        X_train, y_train = oversample(X_train, y_train, RANDOM_STATE)
        print(f"      training rows {before:,} -> {len(X_train):,} (test split untouched)")

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
    }
    pipeline = build_pipeline(numeric, text, categorical,
                              params=params, scale_pos_weight=scale_pos_weight)

    # ---- stage 5: train and evaluate -------------------------------------
    print("[5/6] training XGBoost")
    y_train_int = y_train.map(label_to_int).astype(int)
    pipeline.fit(X_train, y_train_int)

    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pd.Series(pipeline.predict(X_test)).map(int_to_label)
    metrics = evaluation.evaluate(y_test, y_pred, proba,
                                  positive=positive, classes=[negative, positive])
    if args.cv_folds > 1:
        print(f"      cross-validating ({args.cv_folds}-fold)")
        metrics.update(evaluation.cross_val_accuracy(
            build_pipeline(numeric, text, categorical,
                           params=params, scale_pos_weight=scale_pos_weight),
            X, y.map(label_to_int).astype(int),
            folds=args.cv_folds, random_state=RANDOM_STATE,
        ))
    print(evaluation.format_report(metrics))

    warnings = list(args.note) + evaluation.quality_warnings(metrics)
    if avail["unavailable"]:
        warnings.append(
            "Paper Table 1 attributes not present in this dataset and therefore not "
            f"modelled: {', '.join(avail['unavailable'])}. They were left out rather "
            "than synthesised."
        )
    for w in warnings:
        print(f"  ! WARNING: {w}")

    # ---- stage 6: save ----------------------------------------------------
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    importances = pipeline.named_steps["classifier"].feature_importances_
    top = sorted(
        ({"feature": n, "importance": round(float(v), 6)} for n, v in zip(names, importances)),
        key=lambda d: d["importance"], reverse=True,
    )[:25]

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    import xgboost

    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "algorithm": "XGBClassifier + TF-IDF(bio) + OneHot(language) + StandardScaler(numeric)",
        "paper": "Fake Profile Detection Using XGBoost Algorithm (Venkadesh et al., ICRDICCT'25)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "python_version": platform.python_version(),
        "dataset": str(args.dataset),
        "training_samples": int(len(X_train)),
        "classes": [negative, positive],
        "positive_class": positive,
        "label_mapping": label_to_int,
        # The API rebuilds features with the same function and asserts against
        # this list, so a feature added to ml.features without retraining is
        # caught at startup instead of silently shifting every column.
        "feature_columns": list(X.columns),
        # Post-transform width (numerics + TF-IDF + one-hot), which is what
        # GET /api/v1/model/info reports; len(feature_columns) is the pre-
        # transform count and would understate it by ~200.
        "feature_count": len(names),
        "numeric_columns": numeric,
        "text_column": text,
        "categorical_column": categorical,
        "paper_features": avail,
        "cleaning": report.as_dict(),
        "imbalance": {
            "strategy": strategy,
            "reason": why,
            "negatives_per_positive": round(ratio, 4),
            "scale_pos_weight": round(scale_pos_weight, 4) if scale_pos_weight else None,
        },
        "hyperparameters": params,
        "metrics": metrics,
        "top_feature_importances": top,
        "warnings": warnings,
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[6/6] saved {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")
    print(f"      saved {args.out / 'model_meta.json'}")
    print("\n      top features by gain:")
    for row in top[:8]:
        print(f"        {row['feature']:26s} {row['importance']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
