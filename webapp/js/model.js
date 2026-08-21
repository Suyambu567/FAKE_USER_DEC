/**
 * Browser inference for the exported scikit-learn pipeline.
 *
 * This is not a reimplementation of the detector — it is the *same fitted model*,
 * evaluated in JavaScript. The scaler means, the TF-IDF vocabulary and idf
 * weights, and every tree split come straight out of the Python artifact via
 * ml/export_to_json.py. Change the training script and this file needs no edits.
 *
 * Everything runs locally. No network call is made after model.json is fetched,
 * and nothing the user types ever leaves the page.
 */

const Model = (() => {
  'use strict';

  let spec = null;

  // sklearn's default token pattern: runs of two or more word characters.
  const TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu;

  /**
   * Mirror of sklearn's strip_accents='unicode'.
   * NFKD splits "é" into "e" + combining acute; we then drop the combining marks.
   */
  function stripAccents(text) {
    return text.normalize('NFKD').replace(/\p{M}/gu, '');
  }

  /**
   * Replicates TfidfVectorizer.transform for a single document.
   *
   * Terms absent from the fitted vocabulary are dropped, which is exactly how
   * sklearn behaves — and is why an unseen bio can never throw here. The Flask
   * app crashed on precisely this case because it used a LabelEncoder instead.
   */
  function tfidfVector(text, tfidf) {
    const dim = tfidf.idf.length;
    const vec = new Float64Array(dim);
    if (!text) return vec;

    let doc = String(text);
    if (tfidf.lowercase) doc = doc.toLowerCase();
    doc = stripAccents(doc);

    const counts = new Map();
    for (const match of doc.matchAll(TOKEN_RE)) {
      const idx = tfidf.vocabulary[match[0]];
      if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    if (counts.size === 0) return vec;

    // tf * idf, then L2 normalise the row — sklearn's order of operations.
    for (const [idx, count] of counts) {
      const tf = tfidf.sublinear_tf ? 1 + Math.log(count) : count;
      vec[idx] = tf * tfidf.idf[idx];
    }
    if (tfidf.norm === 'l2') {
      let sumSq = 0;
      for (const v of vec) sumSq += v * v;
      const norm = Math.sqrt(sumSq);
      if (norm > 0) for (let i = 0; i < dim; i++) vec[i] /= norm;
    }
    return vec;
  }

  /**
   * Builds the full feature row: standardised numerics followed by TF-IDF,
   * in the same column order the ColumnTransformer produced at fit time.
   */
  function buildRow(profile) {
    const pre = spec.preprocessor;
    const num = pre.numeric;
    const numericLen = num ? num.columns.length : 0;
    const tfidfLen = pre.tfidf ? pre.tfidf.idf.length : 0;
    const row = new Float64Array(numericLen + tfidfLen);

    for (let i = 0; i < numericLen; i++) {
      const raw = Number(profile[num.columns[i]]);
      const value = Number.isFinite(raw) ? raw : 0;
      // StandardScaler: (x - mean) / scale
      row[i] = (value - num.mean[i]) / (num.scale[i] || 1);
    }

    if (pre.tfidf) {
      const text = profile[pre.tfidf.column];
      const tv = tfidfVector(text, pre.tfidf);
      row.set(tv, numericLen);
    }
    return row;
  }

  /** Walks one flattened tree to its leaf and returns that leaf's class probabilities. */
  function walkTree(tree, row) {
    let node = 0;
    // feature === -2 marks a leaf in sklearn's representation.
    while (tree.feature[node] !== -2) {
      node = row[tree.feature[node]] <= tree.threshold[node]
        ? tree.left[node]
        : tree.right[node];
    }
    return tree.value[node];
  }

  /** Class probabilities for one feature row. */
  function probabilities(row) {
    const est = spec.estimator;
    const nClasses = spec.classes.length;

    if (est.type === 'forest') {
      // A RandomForest's predict_proba is the mean of the per-tree leaf
      // distributions — not a majority vote.
      const acc = new Float64Array(nClasses);
      for (const tree of est.trees) {
        const leaf = walkTree(tree, row);
        for (let c = 0; c < nClasses; c++) acc[c] += leaf[c];
      }
      const out = {};
      for (let c = 0; c < nClasses; c++) out[spec.classes[c]] = acc[c] / est.trees.length;
      return out;
    }

    if (est.type === 'linear') {
      let z = est.intercept;
      for (let i = 0; i < est.coef.length; i++) z += est.coef[i] * row[i];
      const p = est.squash === 'sigmoid' ? 1 / (1 + Math.exp(-z)) : (z > 0 ? 1 : 0);
      // coef_[0] scores classes_[1] in sklearn's binary convention.
      return { [spec.classes[0]]: 1 - p, [spec.classes[1]]: p };
    }

    throw new Error(`unsupported estimator type: ${est.type}`);
  }

  return {
    async load(url) {
      const response = await fetch(url, { cache: 'force-cache' });
      if (!response.ok) {
        throw new Error(`Could not load the model (HTTP ${response.status}). ` +
                        'Are you opening the page over file:// instead of a local server?');
      }
      spec = await response.json();
      return spec.meta || {};
    },

    get isLoaded() { return spec !== null; },
    get meta() { return (spec && spec.meta) || {}; },
    get classes() { return spec ? spec.classes.slice() : []; },
    get numericColumns() {
      return spec && spec.preprocessor.numeric
        ? spec.preprocessor.numeric.columns.slice() : [];
    },

    /** Mean of each numeric feature in the training set — the "typical" profile. */
    get numericMeans() {
      const num = spec && spec.preprocessor.numeric;
      if (!num) return {};
      return Object.fromEntries(num.columns.map((c, i) => [c, num.mean[i]]));
    },

    /**
     * Classify one profile.
     * @returns {{label:string, confidence:number, probabilities:Object, latencyMs:number}}
     */
    predict(profile) {
      if (!spec) throw new Error('Model is not loaded yet.');
      const started = performance.now();
      const probs = probabilities(buildRow(profile));

      let label = spec.classes[0];
      for (const c of spec.classes) if (probs[c] > probs[label]) label = c;

      return {
        label,
        confidence: probs[label],
        probabilities: probs,
        latencyMs: performance.now() - started,
      };
    },

    /**
     * Per-field contribution to the verdict, by ablation.
     *
     * For each input field, re-score the profile with that field replaced by its
     * training-set mean (a blank string for the bio) and measure how far the
     * "Fake" probability moves. A large positive delta means this field is what
     * pushed the verdict toward fake.
     *
     * Ablation is used rather than the model's global feature_importances_
     * because importances describe the *model*, while this describes *this
     * profile* — which is what a user asking "why?" actually wants.
     */
    explain(profile, targetClass) {
      if (!spec) throw new Error('Model is not loaded yet.');
      const base = probabilities(buildRow(profile))[targetClass];
      const means = this.numericMeans;
      const contributions = [];

      for (const column of Object.keys(means)) {
        const neutral = { ...profile, [column]: means[column] };
        contributions.push({
          feature: column,
          value: profile[column],
          typical: means[column],
          delta: base - probabilities(buildRow(neutral))[targetClass],
        });
      }

      if (spec.preprocessor.tfidf) {
        const bioCol = spec.preprocessor.tfidf.column;
        const neutral = { ...profile, [bioCol]: '' };
        contributions.push({
          feature: bioCol,
          value: profile[bioCol],
          typical: null,
          delta: base - probabilities(buildRow(neutral))[targetClass],
        });
      }

      return contributions.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    },
  };
})();
