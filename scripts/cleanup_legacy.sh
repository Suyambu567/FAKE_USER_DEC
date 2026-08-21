#!/usr/bin/env bash
#
# Remove dead files identified in docs/AUDIT.md §7.
#
# DRY RUN BY DEFAULT. Nothing is deleted until you pass --apply.
# Review the printed list first -- this removes ~384 MB of the 410 MB project.
#
#   ./scripts/cleanup_legacy.sh              # list what would go
#   ./scripts/cleanup_legacy.sh --archive    # tar everything up first, then list
#   ./scripts/cleanup_legacy.sh --apply      # actually delete
#
# Everything listed here is either a duplicate, a stale backup, a committed log,
# or a one-off script that has already run. None of it is referenced by
# backend/ or by website/app.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY=false
ARCHIVE=false
for arg in "$@"; do
  case "$arg" in
    --apply)   APPLY=true ;;
    --archive) ARCHIVE=true ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# --- the list ---------------------------------------------------------------

TARGETS=(
  # Windows virtualenv -- 12,290 files, wrong platform, wrong Python (3.14)
  ".venv"

  # One-off codemods. All have already run; several are non-idempotent and
  # rewrite templates/analytics.html in place. fix_chart.py contains a NameError.
  "website/fix_chart.py"
  "website/fix_chart2.py"
  "website/fix_chart3.py"
  "website/modify_analytics.py"
  "website/modify_analytics2.py"
  "website/modify_analytics3.py"
  "website/update_charts.py"
  "website/update_charts2.py"

  # Stale copies of source under version control
  "website/app.py.backup"
  "website/app.py.bak"
  "website/templates/analytics.html.backup"
  "website/templates/analytics.html.backup2"
  "website/templates/analytics.html.backup3"
  "website/templates/analytics.html.bak"
  "website/templates/analytics.html.bak2"

  # Scratch fragments left by the codemods
  "website/templates/inline_block.txt"
  "website/templates/inline_block2.txt"
  "website/templates/new_block.txt"
  "website/templates/new_block2.txt"
  "website/templates/prefix.txt"
  "website/templates/prefix2.txt"
  "website/templates/suffix.txt"
  "website/templates/suffix2.txt"

  # Committed runtime logs
  "website/flask2.log"
  "website/flask3.log"
  "website/flask_debug.log"
  "website/flask_debug2.log"
  "website/flask_new.log"

  # Generated HTML dumps
  "website/analytics_output.html"
  "website/page.html"

  # Redundant model artifacts. website/trained_model.pkl is the one the Flask app
  # loads; backend/artifacts/model.joblib is the one the new API loads.
  "trained_model.pkl"                           # 72 MB, root copy
  "website/trained_model.pkl.backup"            # 72 MB
  "website/trained_model_original_copy.pkl"     # 109 MB
  "FAKE_PROFILE_TRAIN_CODE/trained_model.pkl"           # 72 MB
  "FAKE_PROFILE_TRAIN_CODE/trained_model_improved.pkl"  # 72 MB

  # Incoherent encoder -- LabelEncoder over 15 bios feeding a TfidfVectorizer.
  # This is the direct cause of the /predict crash (AUDIT.md 2.1).
  "bio_encoder.pkl"
  "website/bio_encoder.pkl"
  "account_encoder.pkl"
  "website/account_encoder.pkl"
  "FAKE_PROFILE_TRAIN_CODE/account_encoder.pkl"
  "FAKE_PROFILE_TRAIN_CODE/account_encoder_improved.pkl"

  # Duplicate datasets. Keep website/dataset.csv (15k, what the model saw) and
  # website/dataset_original.csv (10k, pre-augmentation).
  #
  # data/dataset.csv and advanced_instagram_fake_real_data_filled_bio.csv used
  # to be listed here and are NOT any more. They are the paper pipeline's
  # control experiment: README.md and docs/PAPER_ALIGNMENT.md both document
  # `ml.train_xgb --dataset ../data/dataset.csv`, which is what demonstrates
  # that the pipeline scores at chance when the labels carry no information.
  # Deleting them turns a documented command into a broken one.
  "papper/dataset.csv"
  "FAKE_PROFILE_TRAIN_CODE/dataset.csv"

  # Duplicate training scripts -- superseded by backend/ml/train.py
  "FAKE_PROFILE_TRAIN_CODE/train_improved.py"
  "FAKE_PROFILE_TRAIN_CODE/debug.py"

  # A zip inside the zip, containing one settings file
  "FAKE_USER_DEC.zip"

  # Contains the single character 💪
  "test.txt"

  # Notebook scratch
  "papper/.ipynb_checkpoints"
)

# --- report -----------------------------------------------------------------

printf '%-58s %10s\n' "PATH" "SIZE"
printf '%.0s-' {1..70}; echo

total=0
present=()
for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then
    bytes=$(du -sb "$t" 2>/dev/null | cut -f1)
    total=$(( total + bytes ))
    present+=("$t")
    printf '%-58s %10s\n' "$t" "$(numfmt --to=iec "$bytes")"
  fi
done

printf '%.0s-' {1..70}; echo
printf '%-58s %10s\n' "TOTAL (${#present[@]} paths)" "$(numfmt --to=iec "$total")"
echo

if [[ ${#present[@]} -eq 0 ]]; then
  echo "Nothing to remove -- already clean."
  exit 0
fi

# --- act --------------------------------------------------------------------

if [[ "$ARCHIVE" == true ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  archive="../FAKE_USER_DEC-legacy-${stamp}.tar.gz"
  echo "archiving to ${archive} ..."
  tar czf "$archive" "${present[@]}"
  echo "archived $(du -h "$archive" | cut -f1)"
  echo
fi

if [[ "$APPLY" != true ]]; then
  cat <<'EOF'
DRY RUN -- nothing was deleted.

  ./scripts/cleanup_legacy.sh --archive --apply    # safest: tar first, then delete
  ./scripts/cleanup_legacy.sh --apply              # delete now

Note: website/ still runs after this (its dataset.csv, templates and
trained_model.pkl are kept) -- but /predict remains broken there, by design.
Use backend/ instead.
EOF
  exit 0
fi

echo "deleting ${#present[@]} paths ..."
for t in "${present[@]}"; do
  rm -rf -- "$t"
  echo "  removed $t"
done
echo
echo "done. freed $(numfmt --to=iec "$total")"
