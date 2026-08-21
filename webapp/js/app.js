/**
 * UI wiring.
 *
 * Holds no detection logic of its own — `Model` owns inference and `Features`
 * owns input handling. This file only reads the form, calls them, and renders.
 */

(() => {
  'use strict';

  const MODEL_URL = 'model/model.json';

  // Below this much lift over the majority-class baseline, the model is not
  // meaningfully better than guessing and the UI says so instead of presenting
  // a percentage as a finding.
  const MIN_USEFUL_LIFT = 0.05;

  const $ = (id) => document.getElementById(id);

  const el = {
    modelStatus: $('modelStatus'),
    form: $('analyzeForm'),
    userId: $('userId'),
    userIdError: $('userIdError'),
    profileDetails: $('profileDetails'),
    profileHint: $('profileHint'),
    analyzeBtn: $('analyzeBtn'),
    btnSpinner: document.querySelector('.spinner-btn'),
    fillDemo: $('fillDemo'),
    clearForm: $('clearForm'),
    result: $('result'),
    demoBanner: $('demoBanner'),
    verdictBadge: $('verdictBadge'),
    verdictIcon: $('verdictIcon'),
    verdictUser: $('verdictUser'),
    verdictLabel: $('verdictLabel'),
    verdictSub: $('verdictSub'),
    confidenceValue: $('confidenceValue'),
    confidenceBar: $('confidenceBar'),
    probBreakdown: $('probBreakdown'),
    reliabilityNote: $('reliabilityNote'),
    reasons: $('reasons'),
    profileDump: $('profileDump'),
    modelCard: $('modelCard'),
    modelMeta: $('modelMeta'),
    modelWarnings: $('modelWarnings'),
    themeToggle: $('themeToggle'),
  };

  const PROFILE_FIELDS = ['followers', 'following', 'posts', 'engagement',
                          'likes', 'comments', 'age', 'bio'];

  let modelUsable = true;

  /* ---- theme ------------------------------------------------------------ */

  function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);

    el.themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
      const next = current
        ? (current === 'dark' ? 'light' : 'dark')
        : (prefersDark ? 'light' : 'dark');
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  }

  /* ---- errors ----------------------------------------------------------- */

  function setFieldError(name, message) {
    const input = $(name);
    const box = document.querySelector(`[data-error-for="${name}"]`);
    if (input) input.setAttribute('aria-invalid', message ? 'true' : 'false');
    if (!box) return;
    box.textContent = message || '';
    box.hidden = !message;
  }

  function clearErrors() {
    setUserIdError('');
    PROFILE_FIELDS.forEach((f) => setFieldError(f, ''));
  }

  function setUserIdError(message) {
    el.userIdError.textContent = message || '';
    el.userIdError.hidden = !message;
    el.userId.setAttribute('aria-invalid', message ? 'true' : 'false');
  }

  /* ---- model load ------------------------------------------------------- */

  async function loadModel() {
    try {
      const meta = await Model.load(MODEL_URL);
      el.modelStatus.hidden = true;
      renderModelCard(meta);
    } catch (error) {
      el.modelStatus.className = 'notice notice-danger';
      el.modelStatus.innerHTML =
        `<span><strong>The model could not be loaded.</strong> ${escapeHtml(error.message)}</span>`;
      el.analyzeBtn.disabled = true;
    }
  }

  function renderModelCard(meta) {
    if (!meta || !Object.keys(meta).length) return;

    const metrics = meta.metrics || {};
    const lift = Number(metrics.lift_over_baseline ?? 0);
    modelUsable = lift >= MIN_USEFUL_LIFT;

    const rows = [
      ['Version', meta.model_version],
      ['Trained', meta.trained_at ? new Date(meta.trained_at).toLocaleString() : null],
      ['Algorithm', meta.algorithm],
      ['Training rows', meta.training_samples ? Number(meta.training_samples).toLocaleString() : null],
      ['Accuracy', metrics.accuracy != null ? pct(metrics.accuracy) : null],
      ['Always-guess baseline', metrics.baseline_accuracy != null ? pct(metrics.baseline_accuracy) : null],
      ['Lift over baseline', metrics.lift_over_baseline != null ? pct(metrics.lift_over_baseline) : null],
      ['ROC-AUC', metrics.roc_auc != null ? metrics.roc_auc.toFixed(3) : null],
    ].filter(([, v]) => v != null && v !== '');

    el.modelMeta.innerHTML = rows
      .map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd></div>`)
      .join('');

    const warnings = meta.warnings || [];
    el.modelWarnings.innerHTML = warnings.length
      ? warnings.map((w) =>
          `<div class="notice notice-warn" style="margin-top:.9rem">
             <span>${escapeHtml(w)}</span></div>`).join('')
      : '';

    el.modelCard.hidden = false;
  }

  /* ---- analyse ---------------------------------------------------------- */

  function anyProfileFieldFilled() {
    return PROFILE_FIELDS.some((f) => String($(f).value || '').trim() !== '');
  }

  async function analyze(event) {
    event.preventDefault();
    clearErrors();

    const check = Features.validateUserId(el.userId.value);
    if (!check.ok) {
      setUserIdError(check.error);
      el.userId.focus();
      return;
    }
    const userId = check.value;

    let profile;
    let isDemo;

    if (anyProfileFieldFilled()) {
      const form = Object.fromEntries(PROFILE_FIELDS.map((f) => [f, $(f).value]));
      form.verified = $('verified').checked;

      const parsed = Features.readForm(form);
      if (!parsed.ok) {
        Object.entries(parsed.errors).forEach(([field, msg]) => setFieldError(field, msg));
        el.profileDetails.open = true;
        $(Object.keys(parsed.errors)[0])?.focus();
        return;
      }
      profile = parsed.profile;
      isDemo = false;
    } else {
      profile = Features.demoProfile(userId);
      isDemo = true;
    }

    setBusy(true);
    // One frame so the spinner paints before the (synchronous) forest walk.
    await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 120)));

    try {
      const prediction = Model.predict(profile);
      const explanation = Model.explain(profile, prediction.label);
      render(userId, profile, prediction, explanation, isDemo);
    } catch (error) {
      el.modelStatus.hidden = false;
      el.modelStatus.className = 'notice notice-danger';
      el.modelStatus.innerHTML =
        `<span><strong>Analysis failed.</strong> ${escapeHtml(error.message)}</span>`;
    } finally {
      setBusy(false);
    }
  }

  function setBusy(busy) {
    el.analyzeBtn.disabled = busy;
    el.btnSpinner.hidden = !busy;
    el.analyzeBtn.querySelector('.btn-label').textContent = busy ? 'Analyzing…' : 'Analyze';
  }

  /* ---- render ----------------------------------------------------------- */

  function render(userId, profile, prediction, explanation, isDemo) {
    const isFake = prediction.label === 'Fake';

    el.demoBanner.hidden = !isDemo;

    el.verdictUser.textContent = userId;
    el.verdictIcon.textContent = isFake ? '🚨' : '✅';
    el.verdictLabel.textContent = isFake ? 'Fake User' : 'Real User';
    el.verdictBadge.classList.toggle('is-fake', isFake);
    el.verdictLabel.classList.toggle('is-fake', isFake);
    el.verdictSub.textContent =
      `Scored in ${prediction.latencyMs.toFixed(1)} ms, entirely in your browser.`;

    const confidence = prediction.confidence;
    el.confidenceValue.textContent = pct(confidence);
    el.confidenceBar.style.width = `${(confidence * 100).toFixed(1)}%`;
    el.confidenceBar.classList.toggle('is-fake', isFake);

    el.probBreakdown.innerHTML = Object.entries(prediction.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(([cls, p]) => `<span>${escapeHtml(cls)}: <strong>${pct(p)}</strong></span>`)
      .join('');

    renderReliability(confidence, isDemo);
    renderReasons(explanation, prediction.label);
    renderProfile(profile);

    el.result.hidden = false;
    el.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderReliability(confidence, isDemo) {
    const notes = [];

    if (!modelUsable) {
      notes.push('<strong>This model is not decision-grade.</strong> Its accuracy is ' +
        'barely above always guessing the most common class, so this verdict carries ' +
        'almost no information. Retrain on labelled real-world data before relying on it.');
    }
    // A binary classifier at ~50% is expressing no opinion, whatever the label says.
    if (confidence < 0.6) {
      notes.push('<strong>Low confidence.</strong> The model is close to undecided ' +
        'between the two classes here; treat this as "unknown", not as a verdict.');
    }
    if (isDemo && notes.length === 0) {
      notes.push('<strong>Demo input.</strong> The verdict is real for the numbers shown, ' +
        'but those numbers were generated, not measured.');
    }

    el.reliabilityNote.innerHTML = notes.map((n) => `<span>${n}</span>`).join('<br><br>');
    el.reliabilityNote.hidden = notes.length === 0;
  }

  function renderReasons(explanation, label) {
    // Ablations under half a point move nothing a user should read into.
    const meaningful = explanation.filter((r) => Math.abs(r.delta) >= 0.005).slice(0, 6);

    if (!meaningful.length) {
      el.reasons.innerHTML =
        '<li class="reasons-empty">No single factor moved this verdict much — ' +
        'the model reached it from the combination of all of them.</li>';
      return;
    }

    el.reasons.innerHTML = meaningful.map((r) => {
      const towardLabel = r.delta > 0;
      const pushesFake = (label === 'Fake') === towardLabel;
      const shown = Features.format(r.feature, r.value);
      const typical = r.typical == null ? null : Features.format(r.feature, r.typical);

      return `
        <li class="reason ${pushesFake ? 'pushes-fake' : 'pushes-real'}">
          <span class="reason-name">${escapeHtml(Features.label(r.feature))}</span>
          <span class="reason-detail">${escapeHtml(shown)}${
            typical ? ` &middot; typical ${escapeHtml(typical)}` : ''}</span>
          <span class="reason-impact">${towardLabel ? '↑' : '↓'} ${
            (Math.abs(r.delta) * 100).toFixed(1)} pts</span>
        </li>`;
    }).join('');
  }

  function renderProfile(profile) {
    el.profileDump.innerHTML = Object.entries(profile).map(([column, value]) => {
      const isBio = column === Features.COLUMNS.bio;
      return `<div><dt>${escapeHtml(Features.label(column))}</dt>
              <dd${isBio ? ' class="mono"' : ''}>${
                escapeHtml(Features.format(column, value))}</dd></div>`;
    }).join('');
  }

  /* ---- helpers ---------------------------------------------------------- */

  const pct = (v) => `${(Number(v) * 100).toFixed(1)}%`;

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text ?? '');
    return div.innerHTML;
  }

  function fillDemoForm() {
    const check = Features.validateUserId(el.userId.value);
    const userId = check.ok ? check.value : 'demo_user';
    if (!check.ok) el.userId.value = userId;

    const p = Features.demoProfile(userId);
    const C = Features.COLUMNS;
    $('followers').value  = p[C.followers];
    $('following').value  = p[C.following];
    $('posts').value      = p[C.posts];
    $('engagement').value = p[C.engagement];
    $('likes').value      = p[C.likes];
    $('comments').value   = p[C.comments];
    $('age').value        = p[C.age];
    $('bio').value        = p[C.bio];
    $('verified').checked = p[C.verified] === 1;
    clearErrors();
    updateHint();
  }

  function clearProfileForm() {
    PROFILE_FIELDS.forEach((f) => { $(f).value = ''; });
    $('verified').checked = false;
    clearErrors();
    updateHint();
  }

  function updateHint() {
    el.profileHint.textContent = anyProfileFieldFilled()
      ? 'using the values you entered'
      : 'optional — demo data is used if left blank';
  }

  /* ---- init ------------------------------------------------------------- */

  initTheme();
  el.form.addEventListener('submit', analyze);
  el.fillDemo.addEventListener('click', fillDemoForm);
  el.clearForm.addEventListener('click', clearProfileForm);
  PROFILE_FIELDS.forEach((f) => $(f).addEventListener('input', updateHint));
  el.userId.addEventListener('input', () => setUserIdError(''));

  loadModel();
})();
