# Which ML algorithm is best for this problem?

14 algorithms, cross-validated, on two datasets. Reproduce with
`backend/ml/benchmark.py`; raw tables in `benchmark_shipped.md` and
`benchmark_signal.md`, raw numbers in the matching `.json` files.

**Short answer:** on the dataset shipped with this project, none of them —
every algorithm scores at chance, because the labels are random. On data that
has signal, **Random Forest** wins on ROC-AUC, but its margin over a plain
Logistic Regression is 0.4 points and smaller than the fold-to-fold variation.
The decision is therefore made on cost, not accuracy.

---

## 1. Method

| | |
|---|---|
| Validation | Stratified k-fold cross-validation (5-fold shipped, 3-fold signal) |
| Ranking metric | **ROC-AUC** — threshold-independent, so a model is not rewarded for landing on a lucky operating point |
| Secondary | **PR-AUC** (average precision) — the metric that matters at a realistic 5% fake rate, where ROC-AUC stays flattering |
| Reported alongside | Accuracy **and** the majority-class baseline. Accuracy alone is meaningless on imbalanced data; the gap is the only interesting number |
| Cost | Mean fit seconds per fold, inference ms per 1,000 rows, serialised artifact size |
| Preprocessing | Identical for every algorithm: `StandardScaler` on 8 numerics + `TfidfVectorizer(max_features=120)` on bio text |
| Positive class | `Fake` |

Every model got the same preprocessing and the same folds, so differences are
attributable to the estimator.

---

## 2. The shipped dataset: nothing works

`website/dataset.csv`, 15,000 rows sampled, 5-fold CV, baseline accuracy 0.5081.

| Model | Accuracy | ROC-AUC | PR-AUC | Fit (s) | Lift |
|---|---|---|---|---|---|
| k-Nearest Neighbours (k=25) | 0.5097 | **0.5124** | 0.5163 | 0.12 | +0.0016 |
| Gaussian Naive Bayes | 0.5121 | 0.5124 | 0.5156 | 0.14 | +0.0040 |
| Linear SVM | 0.5087 | 0.5110 | 0.5165 | 0.13 | +0.0006 |
| Logistic Regression | 0.5090 | 0.5110 | 0.5166 | 0.22 | +0.0009 |
| HistGradientBoosting | 0.5093 | 0.5102 | 0.5150 | 0.29 | +0.0012 |
| AdaBoost | 0.5083 | 0.5092 | 0.5148 | 2.69 | +0.0002 |
| MLP (128,64) | 0.5089 | 0.5087 | 0.5153 | 1.83 | +0.0008 |
| Gradient Boosting | 0.5029 | 0.5082 | 0.5136 | 8.06 | −0.0052 |
| Random Forest | 0.5053 | 0.5070 | 0.5134 | 0.94 | −0.0028 |
| Extra Trees | 0.5061 | 0.5064 | 0.5099 | 0.77 | −0.0020 |
| XGBoost | 0.5027 | 0.5055 | 0.5165 | 241.16 | −0.0054 |
| Decision Tree (depth 8) | 0.4988 | 0.5013 | 0.5087 | 0.24 | −0.0093 |
| *Baseline (majority class)* | *0.5081* | *0.5000* | *0.5081* | — | *0.0000* |
| LightGBM | 0.4971 | **0.4966** | 0.5084 | 1253.11 | −0.0110 |

**Chance is ROC-AUC 0.5. The best result is 0.5124.**

Read the bottom row carefully: LightGBM spent **1,253 seconds per fold** and
finished at ROC-AUC 0.4966 — *below chance*. XGBoost spent 241 seconds to reach
0.5055. Six of the fourteen have negative lift: they are worse than a constant
prediction.

This is not a tuning problem and no hyperparameter search fixes it. `data/app.py`
assigns the label with `np.random.choice(['Real','Fake'])`, independent of every
feature. The largest standardised mean difference between the classes across all
eight numerics is 0.0395 — sampling noise at n=15,000. There is nothing to learn,
so the models correctly learn nothing.

> **The single most expensive mistake available here is spending weeks on model
> selection when the data has no signal.** The benchmark costs 20 minutes and
> rules it out definitively.

---

## 3. A dataset with signal: everything works, and they tie

`data/synthetic_signal.csv` (from `backend/ml/generate_dataset.py`), 20,000 rows
sampled, 3-fold CV, baseline accuracy 0.6260, 6% label noise.

Sorted by ROC-AUC:

| Model | Accuracy | ±std | ROC-AUC | PR-AUC | Fit (s) | Predict (ms/1k) | Size (MB) |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.9374 | 0.0014 | **0.9320** | **0.9064** | 1.84 | 48.0 | 2.80 |
| AdaBoost | 0.9283 | 0.0060 | 0.9303 | 0.9015 | 5.18 | 60.6 | 0.02 |
| k-Nearest Neighbours (k=25) | 0.9378 | 0.0014 | 0.9295 | 0.8980 | 0.19 | **616.5** | 1.32 |
| Gradient Boosting | 0.9356 | 0.0014 | 0.9291 | 0.8992 | 14.08 | 16.9 | 0.15 |
| XGBoost | 0.9371 | 0.0013 | 0.9288 | 0.8992 | 186.36 | 17.7 | 0.57 |
| Extra Trees | **0.9384** | 0.0015 | 0.9287 | 0.8985 | 1.65 | 50.8 | 1.22 |
| LightGBM | 0.9365 | 0.0014 | 0.9287 | 0.8984 | **1386.37** | 13.7 | 1.22 |
| Logistic Regression | 0.9383 | 0.0014 | 0.9282 | 0.8970 | 2.00 | 16.2 | **0.00** |
| HistGradientBoosting | 0.9368 | 0.0014 | 0.9282 | 0.8980 | 183.47 | 137.3 | 0.11 |
| Gaussian Naive Bayes | 0.9380 | 0.0015 | 0.9282 | 0.8987 | **0.20** | 13.4 | 0.01 |
| Linear SVM | **0.9384** | 0.0015 | 0.9279 | 0.8986 | 0.36 | 17.5 | **0.00** |
| MLP (128,64) | 0.9383 | 0.0014 | 0.9274 | 0.8958 | 12.95 | 18.1 | 0.53 |
| Decision Tree (depth 8) | 0.9314 | 0.0028 | 0.9260 | 0.8857 | 0.33 | 12.3 | 0.01 |
| *Baseline (majority class)* | *0.6260* | *0.0000* | *0.5000* | *0.3740* | — | — | — |

Every algorithm clears the baseline by ~31 points. And they are **statistically
indistinguishable from one another**: ROC-AUC spans 0.9260 to 0.9320 — a 0.6
point range — while the fold-to-fold accuracy standard deviation is ±0.0014.
Ranking Extra Trees above Logistic Regression on 0.0001 of accuracy is noise, not
a finding.

### Why they converge

The models are at the ceiling. The generator flips 6% of labels outright, which
caps any classifier at **94%** accuracy no matter how good it is. The best
observed accuracy is 0.9384 — **within 0.16 points of the theoretical maximum**.

When a problem's signal is strong and its irreducible error dominates, algorithm
choice stops being a lever. Everything that can represent the decision boundary
reaches the same place.

### Feature engineering does not rescue it either

The obvious next move is derived features — the discriminative signal here is
relational (following-per-follower, comments-per-like, posts-per-year), and a
linear model cannot express a ratio at all while a tree can only approximate one
with a staircase of axis-aligned splits. Measured:

| Setup | ROC-AUC |
|---|---|
| Logistic Regression, base features | 0.9282 |
| Logistic Regression, **+ derived ratios** | 0.9311 (+0.0029) |
| Random Forest, base features | 0.9320 |
| Random Forest, **+ derived ratios** | 0.9306 (−0.0014) |

Derived features buy Logistic Regression +0.3 points — more than the best
algorithm swap gives — and cost Random Forest 0.1. Both are inside the noise
band. At the ceiling, nothing helps.

**This is the real lesson, and it generalises:** measure the irreducible error
first. If you are within a fraction of a point of it, further modelling work has
no headroom, and effort belongs on data quality instead.

---

## 4. The recommendation

Accuracy does not separate these models, so choose on cost and constraints.

### For this project's browser app — **Random Forest**

* Best ROC-AUC (0.9320) and best PR-AUC (0.9064), which is the metric that
  matters if the real fake rate is low.
* Exports cleanly to JSON for client-side inference — the trees flatten into
  parallel arrays (`ml/export_to_json.py`). XGBoost, LightGBM and
  HistGradientBoosting have no exporter here; their tree formats differ.
* Bounded to 60 trees at depth 9 it costs **256 KB** (70 KB gzipped) and scores
  0.9342. The unbounded 200-tree version is 2.80 MB — a slow first load on
  mobile data for no measurable gain.

### For a cost-sensitive server — **Logistic Regression** or **Gaussian Naive Bayes**

0.4 points of ROC-AUC behind Random Forest, and:

* Logistic Regression: **0.00 MB** artifact (nine coefficients), 2.0 s to fit,
  16 ms per 1,000 predictions, and coefficients that are directly readable as
  explanations.
* Gaussian Naive Bayes: fastest fit in the field at **0.20 s**, 0.01 MB.

If the model must be retrained hourly, or explained to a non-technical
stakeholder, these are the right answer and the 0.4 points is a rounding error.

### Avoid

| Model | Why |
|---|---|
| **k-Nearest Neighbours** | 616 ms per 1,000 predictions — **36x slower** than Logistic Regression at inference, because it has no model: every prediction scans the training set. Fine offline, disqualifying in a request path. |
| **LightGBM** | 1,386 s per fold on 20k rows for ROC-AUC 0.9287 — *below* Random Forest's 0.9320 at 1.84 s. It is built for datasets orders of magnitude larger; here the overhead is pure cost. |
| **XGBoost** | 186 s per fold for 0.9288. Same story. |
| **HistGradientBoosting** | 183 s to fit *and* 137 ms/1k to predict — the worst of both. |
| **Decision Tree** | Lowest ROC-AUC (0.9260) and highest variance (±0.0028, double the field). A single tree is a high-variance estimator; that is exactly what ensembles fix. |

### Do not use accuracy to choose

On the shipped dataset the majority-class baseline scores **0.5081 accuracy and
an F1 of 0.6738** by always answering "Fake" — it looks like a working model on
those two numbers alone. Its ROC-AUC is 0.5000, which is what gives it away.
Report accuracy next to its baseline, and rank on ROC-AUC or PR-AUC.

---

## 5. What actually matters, ranked

1. **Real labelled data.** No algorithm recovers signal that is not present. The
   shipped dataset caps every model at chance; this is not fixable downstream.
2. **Knowing your irreducible error.** The signal dataset caps everything at 94%.
   Models reach 93.84%. Further modelling work has 0.16 points of headroom —
   effort belongs on label quality, not on architecture.
3. **The right metric.** PR-AUC at a realistic 5% fake rate, not accuracy.
4. **Deployment constraints.** Artifact size, inference latency and
   exportability decided this project's model. Accuracy did not, because it
   could not.
5. **The algorithm.** Last, and on this evidence worth 0.6 points.

---

## Reproducing

```bash
cd backend
.venv/bin/python -m ml.generate_dataset --rows 50000 --out ../data/synthetic_signal.csv

.venv/bin/python -m ml.benchmark --dataset ../website/dataset.csv --folds 5 --sample 15000 \
    --out ../docs/benchmark_shipped.md --json-out ../docs/benchmark_shipped.json

.venv/bin/python -m ml.benchmark --dataset ../data/synthetic_signal.csv --folds 3 --sample 20000 \
    --out ../docs/benchmark_signal.md --json-out ../docs/benchmark_signal.json

.venv/bin/python -m ml.benchmark --dataset ../data/synthetic_signal.csv --derived --folds 3
```

Runtime on an 80-core host: ~20 min for the shipped set (LightGBM and XGBoost
dominate), ~45 min for the signal set. Drop `LightGBM`/`XGBoost` from
`ml/benchmark.py::candidates()` for a two-minute sweep.
