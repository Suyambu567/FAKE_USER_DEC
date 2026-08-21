#!/usr/bin/env python3
"""Benchmark classification algorithms on the fake-profile task.

Answers "which algorithm is best?" with numbers rather than folklore, and — just
as importantly — shows when the answer is "none of them, fix your data".

What is measured, and why each one is here:

*   **ROC-AUC** — the primary ranking metric. Threshold-independent, so it does
    not reward a model for happening to sit at a lucky operating point.
*   **PR-AUC (average precision)** — the metric that matters for fraud. At a
    realistic 5% fake rate, ROC-AUC stays flattering while PR-AUC collapses.
*   **Accuracy next to the majority-class baseline** — accuracy alone is
    meaningless on imbalanced data. The gap is the only interesting number.
*   **Fit time and inference latency** — a model 0.4 points better that is 50x
    slower to serve is usually the wrong trade at 10k+ users.
*   **Artifact size** — resident memory per worker, multiplied by worker count.

Every score is cross-validated (stratified k-fold), so a single lucky split
cannot crown a winner. Standard deviation across folds is reported: two models
whose means differ by less than a fold-to-fold std are tied, not ranked.

Usage:
    python -m ml.benchmark --dataset ../website/dataset.csv           # shipped data
    python -m ml.benchmark --dataset ../data/synthetic_signal.csv     # with signal
    python -m ml.benchmark --dataset <csv> --derived                  # + ratio features
    python -m ml.benchmark --dataset <csv> --out ../docs/benchmark.md
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

TARGET = "Account Type"
TEXT_COL = "Bio Text"
BASE_NUMERIC = [
    "Followers", "Following", "Posts", "Engagement Rate (%)",
    "Avg Likes per Post", "Avg Comments per Post", "Verified",
    "Account Age (Years)",
]
SEED = 42


# ---- derived features ------------------------------------------------------


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Ratios and log transforms the raw columns do not express.

    The discriminative signal in this problem is almost entirely *relational*:
    following-per-follower, comments-per-like, posts-per-year. A tree can only
    approximate a ratio with a staircase of axis-aligned splits, and a linear
    model cannot represent one at all. Computing them explicitly is usually worth
    more than any change of algorithm — which the benchmark demonstrates.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        eps = 1.0
        followers = out["Followers"].astype(float)
        likes = out["Avg Likes per Post"].astype(float)

        # Mass-follow farming: fakes follow far more than they are followed.
        out["ratio_following_followers"] = out["Following"] / (followers + eps)
        # Bots like; bots do not comment. Strongest single signal in the data.
        out["ratio_comments_likes"] = out["Avg Comments per Post"] / (likes + eps)
        # Does the like count match the audience size?
        out["ratio_likes_followers"] = likes / (followers + eps)
        # Posting cadence: real accounts accumulate posts over years.
        out["posts_per_year"] = out["Posts"] / (out["Account Age (Years)"] + 0.1)
        out["followers_per_year"] = followers / (out["Account Age (Years)"] + 0.1)
        # Heavy-tailed counts: log makes them usable by linear/distance models.
        for col in ["Followers", "Following", "Posts", "Avg Likes per Post"]:
            out[f"log_{col}"] = np.log1p(out[col].astype(float))
        # Does the stated engagement rate agree with the raw counts?
        implied = (likes + out["Avg Comments per Post"]) / (followers + eps) * 100
        out["engagement_mismatch"] = (out["Engagement Rate (%)"] - implied).abs()

        return out.replace([np.inf, -np.inf], 0).fillna(0)


DERIVED_COLS = [
    "ratio_following_followers", "ratio_comments_likes", "ratio_likes_followers",
    "posts_per_year", "followers_per_year", "log_Followers", "log_Following",
    "log_Posts", "log_Avg Likes per Post", "engagement_mismatch",
]


def build_preprocessor(numeric_cols: list[str], max_tfidf: int,
                       text_col: str = TEXT_COL) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("txt", TfidfVectorizer(max_features=max_tfidf, stop_words="english",
                                    lowercase=True, strip_accents="unicode", min_df=2),
             text_col),
        ],
        remainder="drop",
        # Dense output: HistGradientBoosting, GaussianNB and MLP cannot take sparse.
        sparse_threshold=0.0,
    )


# ---- the field -------------------------------------------------------------


def candidates() -> dict[str, BaseEstimator]:
    models: dict[str, BaseEstimator] = {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=SEED),
        "Gaussian Naive Bayes": GaussianNB(),
        "k-Nearest Neighbours (k=25)": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000),
        "Decision Tree (depth 8)": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                                          class_weight="balanced",
                                                          random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=14,
                                                min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=14,
                                            min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                        random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, max_depth=8,
                                                               learning_rate=0.1,
                                                               random_state=SEED),
        "MLP (128,64)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120,
                                      early_stopping=True, random_state=SEED),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=7, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
            tree_method="hist", eval_metric="logloss",
            n_jobs=-1, random_state=SEED,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1,
        )
    except ImportError:
        pass

    return models


def score_positive(estimator, X) -> np.ndarray:
    """Probability of the positive class, or a decision score for models without one."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)   # LinearSVC


def evaluate(name: str, estimator, X: pd.DataFrame, y: np.ndarray,
             preprocessor_factory, folds: int) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    acc, roc, prauc, f1s, prec, rec = [], [], [], [], [], []
    fit_times, predict_times = [], []

    for train_idx, test_idx in cv.split(X, y):
        pipe = Pipeline([("prep", preprocessor_factory()), ("clf", estimator)])

        t0 = time.perf_counter()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        fit_times.append(time.perf_counter() - t0)

        Xte = X.iloc[test_idx]
        t0 = time.perf_counter()
        pred = pipe.predict(Xte)
        predict_times.append((time.perf_counter() - t0) / len(test_idx) * 1000)

        yte = y[test_idx]
        acc.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, zero_division=0))
        prec.append(precision_score(yte, pred, zero_division=0))
        rec.append(recall_score(yte, pred, zero_division=0))
        try:
            s = score_positive(pipe, Xte)
            roc.append(roc_auc_score(yte, s))
            prauc.append(average_precision_score(yte, s))
        except Exception:
            roc.append(float("nan"))
            prauc.append(float("nan"))

    # Artifact size from one final full fit.
    pipe = Pipeline([("prep", preprocessor_factory()), ("clf", estimator)])
    pipe.fit(X, y)
    buf = Path("/tmp/_bench_model.joblib")
    joblib.dump(pipe, buf, compress=3)
    size_mb = buf.stat().st_size / 1e6
    buf.unlink(missing_ok=True)

    return {
        "model": name,
        "accuracy": round(float(np.mean(acc)), 4),
        "accuracy_std": round(float(np.std(acc)), 4),
        "roc_auc": round(float(np.nanmean(roc)), 4),
        "pr_auc": round(float(np.nanmean(prauc)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "precision": round(float(np.mean(prec)), 4),
        "recall": round(float(np.mean(rec)), 4),
        "fit_seconds": round(float(np.mean(fit_times)), 2),
        "predict_ms_per_1k": round(float(np.mean(predict_times)) * 1000, 2),
        "artifact_mb": round(size_mb, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-tfidf", type=int, default=120)
    ap.add_argument("--sample", type=int, default=0, help="Subsample rows (0 = all)")
    ap.add_argument("--skip", action="append", default=[], metavar="MODEL",
                    help="Skip a candidate by name, repeatable. LightGBM is the "
                         "usual one: on a many-core host it spends minutes in "
                         "thread contention on data this small, while every other "
                         "candidate finishes in seconds, and it is not one of the "
                         "algorithms the paper compares.")
    ap.add_argument("--paper-features", action="store_true",
                    help="Build features with ml.features.build_feature_frame -- the "
                         "paper's Table 1 profile features plus the behavioural and "
                         "textual groups, i.e. exactly what the deployed XGBoost model "
                         "is trained and served on. Without this the benchmark uses the "
                         "original 8 raw columns, which answers a different question.")
    ap.add_argument("--derived", action="store_true",
                    help="Add ratio/log features before training")
    ap.add_argument("--out", type=Path, help="Write a markdown report here")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.paper_features:
        # Same cleaning and the same feature function the deployed model uses,
        # so "which algorithm is best" is answered on the features that are
        # actually served rather than on a second, benchmark-only feature set.
        from ml import features as ml_features
        from ml.preprocess import clean

        df, report = clean(pd.read_csv(args.dataset))
        if args.sample and args.sample < len(df):
            df = df.sample(args.sample, random_state=SEED).reset_index(drop=True)
        built = ml_features.build_feature_frame(df)
        numeric_cols, text_col, _cat = ml_features.split_columns(built)
        # The categorical branch is dropped here: this compares algorithms, and
        # one-hot columns behave very differently across linear/distance models.
        X = built[[*numeric_cols] + ([text_col] if text_col else [])]
        text_column = text_col or TEXT_COL
        print(f"cleaning: {report.summary()}")
    else:
        df = pd.read_csv(args.dataset).dropna(subset=[*BASE_NUMERIC, TEXT_COL, TARGET])
        if args.sample and args.sample < len(df):
            df = df.sample(args.sample, random_state=SEED)
        df[TEXT_COL] = df[TEXT_COL].astype(str)

        X = df[[*BASE_NUMERIC, TEXT_COL]]
        numeric_cols = list(BASE_NUMERIC)
        if args.derived:
            X = DerivedFeatures().transform(X)
            numeric_cols = BASE_NUMERIC + DERIVED_COLS
        text_column = TEXT_COL

    # Positive class = Fake (the thing we are trying to catch).
    y = (df[TARGET].astype(str) == "Fake").to_numpy().astype(int)
    baseline = float(max(y.mean(), 1 - y.mean()))

    print(f"dataset : {args.dataset}")
    print(f"rows    : {len(df):,}   fake={y.mean():.1%}   baseline accuracy={baseline:.4f}")
    print(f"features: {len(numeric_cols)} numeric + TF-IDF(<= {args.max_tfidf})"
          f"{'  [paper features]' if args.paper_features else ''}"
          f"{'  [derived ON]' if args.derived else ''}")
    print(f"cv      : {args.folds}-fold stratified\n")

    def factory():
        return build_preprocessor(numeric_cols, args.max_tfidf, text_col=text_column)

    rows = []
    skipped = {s.strip().lower() for s in args.skip}
    for name, est in candidates().items():
        if name.strip().lower() in skipped:
            # Announced, never silent: a benchmark that quietly drops a
            # candidate reads as "we compared everything" when it did not.
            print(f"  SKIPPED {name}  (--skip)")
            continue
        print(f"  running {name:32s}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            rows.append(evaluate(name, est, X, y, factory, args.folds))
            r = rows[-1]
            print(f" acc={r['accuracy']:.4f}  roc={r['roc_auc']:.4f}  "
                  f"pr={r['pr_auc']:.4f}  ({time.perf_counter()-t0:.0f}s)")
        except Exception as exc:
            print(f" FAILED: {type(exc).__name__}: {str(exc)[:70]}")

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results["lift"] = (results["accuracy"] - baseline).round(4)

    print("\n" + "=" * 118)
    print(results.to_string(index=False))
    print("=" * 118)

    best = results.iloc[0]
    if best["roc_auc"] < 0.55:
        print("\nVERDICT: no algorithm separates the classes (best ROC-AUC "
              f"{best['roc_auc']:.4f}, chance = 0.5).")
        print("The features carry no information about the label. This is a DATA")
        print("problem; changing the model cannot fix it.")
    else:
        print(f"\nVERDICT: best by ROC-AUC is {best['model']} ({best['roc_auc']:.4f}), "
              f"accuracy {best['accuracy']:.4f} vs {baseline:.4f} baseline.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "dataset": str(args.dataset), "rows": len(df),
            "baseline_accuracy": round(baseline, 4), "folds": args.folds,
            "derived_features": args.derived,
            "results": results.to_dict(orient="records"),
        }, indent=2))
        print(f"\nwrote {args.json_out}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            f"# Benchmark — {args.dataset.name}\n\n"
            f"- rows: {len(df):,}\n- fake rate: {y.mean():.1%}\n"
            f"- baseline accuracy: {baseline:.4f}\n- CV: {args.folds}-fold stratified\n"
            f"- derived features: {'yes' if args.derived else 'no'}\n\n"
            + results.to_markdown(index=False) + "\n"
        )
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
