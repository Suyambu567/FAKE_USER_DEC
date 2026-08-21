#!/usr/bin/env python3
"""Export a fitted sklearn pipeline to plain JSON for browser inference.

The web app has no backend, so a Python pickle is useless to it. Rather than
reimplement or retrain anything, this serialises the *already fitted* pipeline —
the same scaler means, the same TF-IDF vocabulary and idf weights, the same tree
splits — into JSON that `webapp/js/model.js` evaluates directly. The maths in the
browser is the maths sklearn ran; only the language changes.

Supported final estimators:
  * RandomForestClassifier / ExtraTreesClassifier / DecisionTreeClassifier
  * LogisticRegression / LinearSVC

Usage:
    python -m ml.export_to_json --model artifacts/model.joblib \
                                --out ../webapp/model/model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def export_tree(tree, n_classes: int) -> dict:
    """Flatten one sklearn tree into parallel arrays.

    Parallel arrays rather than nested objects: the JSON is ~40% smaller and the
    JS walk is a tight loop over typed arrays instead of pointer chasing.
    """
    t = tree.tree_
    # value[node] is the class distribution at that node; normalise to a
    # probability so the browser can average trees directly.
    values = t.value.reshape(t.node_count, -1)[:, :n_classes]
    totals = values.sum(axis=1, keepdims=True)
    probs = np.divide(values, totals, out=np.full_like(values, 1.0 / n_classes),
                      where=totals > 0)

    return {
        "feature": [int(f) for f in t.feature],          # -2 marks a leaf
        "threshold": [round(float(v), 6) for v in t.threshold],
        "left": [int(v) for v in t.children_left],
        "right": [int(v) for v in t.children_right],
        # Only leaves need a probability; interior nodes get null and compress away.
        "value": [
            [round(float(p), 5) for p in probs[i]] if t.children_left[i] == -1 else None
            for i in range(t.node_count)
        ],
    }


def export_preprocessor(pre) -> dict:
    """Scaler statistics, TF-IDF vocabulary and the output column order."""
    out: dict = {"numeric": None, "tfidf": None, "columns": []}

    for name, transformer, columns in pre.transformers_:
        if name == "num":
            scaler = transformer[-1] if hasattr(transformer, "steps") else transformer
            out["numeric"] = {
                "columns": list(columns),
                "mean": [round(float(v), 8) for v in scaler.mean_],
                "scale": [round(float(v), 8) for v in scaler.scale_],
            }
            out["columns"].extend(columns)

        elif name == "txt":
            vec = transformer[-1] if hasattr(transformer, "steps") else transformer
            vocab = {term: int(idx) for term, idx in vec.vocabulary_.items()}
            out["tfidf"] = {
                "column": columns if isinstance(columns, str) else columns[0],
                "vocabulary": vocab,
                "idf": [round(float(v), 6) for v in vec.idf_],
                "lowercase": bool(vec.lowercase),
                # sklearn's default token pattern; mirrored exactly in model.js.
                "token_pattern": r"(?u)\b\w\w+\b",
                "sublinear_tf": bool(getattr(vec, "sublinear_tf", False)),
                "norm": vec.norm,
            }
            out["columns"].extend(f"tfidf::{t}" for t, _ in
                                  sorted(vocab.items(), key=lambda kv: kv[1]))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--meta", type=Path, help="model_meta.json to embed (metrics, warnings)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pipeline = joblib.load(args.model)
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    classes = [str(c) for c in clf.classes_]

    spec: dict = {
        "format_version": 1,
        "classes": classes,
        "preprocessor": export_preprocessor(pre),
    }

    if isinstance(clf, (RandomForestClassifier, ExtraTreesClassifier)):
        spec["estimator"] = {
            "type": "forest",
            "trees": [export_tree(e, len(classes)) for e in clf.estimators_],
        }
        n_nodes = sum(e.tree_.node_count for e in clf.estimators_)
        detail = f"{len(clf.estimators_)} trees, {n_nodes:,} nodes"

    elif isinstance(clf, DecisionTreeClassifier):
        spec["estimator"] = {"type": "forest", "trees": [export_tree(clf, len(classes))]}
        detail = f"1 tree, {clf.tree_.node_count:,} nodes"

    elif isinstance(clf, (LogisticRegression, LinearSVC)):
        spec["estimator"] = {
            "type": "linear",
            "coef": [round(float(v), 8) for v in clf.coef_[0]],
            "intercept": round(float(clf.intercept_[0]), 8),
            # LinearSVC has no probability; the browser reports a decision score.
            "squash": "sigmoid" if isinstance(clf, LogisticRegression) else "none",
        }
        detail = f"{len(clf.coef_[0])} coefficients"

    else:
        raise SystemExit(
            f"error: {type(clf).__name__} cannot be exported to JSON.\n"
            "Supported: RandomForest, ExtraTrees, DecisionTree, LogisticRegression, "
            "LinearSVC. Retrain with one of these, or serve the model from a backend."
        )

    if args.meta and args.meta.exists():
        meta = json.loads(args.meta.read_text())
        spec["meta"] = {
            "model_version": meta.get("model_version"),
            "trained_at": meta.get("trained_at"),
            "algorithm": meta.get("algorithm"),
            "training_samples": meta.get("training_samples"),
            "dataset": meta.get("dataset"),
            "metrics": meta.get("metrics"),
            # Carried through so the UI can show the honest caveats.
            "warnings": meta.get("warnings", []),
            "feature_importances": meta.get("top_feature_importances", [])[:12],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, separators=(",", ":")))

    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {type(clf).__name__}: {detail}")
    print(f"wrote {args.out}  ({size_mb:.2f} MB)")
    if size_mb > 8:
        print("  ! WARNING: over 8 MB. That is a slow first load on mobile data.")
        print("    Retrain with fewer/shallower trees, or gzip it at the web server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
