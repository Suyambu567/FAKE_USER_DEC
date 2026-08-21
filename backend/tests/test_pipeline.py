"""Tests for the paper pipeline: cleaning, features, imbalance, and skew.

The one that matters most is `test_api_features_match_training_features`. Every
other test here checks a stage in isolation; that one checks the join between
training and serving, which is where this project's predecessor actually broke.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LOG_JSON", "false")

from ml import features  # noqa: E402
from ml.preprocess import clean, imbalance_ratio  # noqa: E402
from ml.train_xgb import choose_strategy, oversample  # noqa: E402

PROFILE = {
    "Real Name": "Priya Kumar",
    "Profile Picture": 1,
    "Profile Link": 0,
    "Language": "English",
    "Followers": 5200,
    "Following": 410,
    "Posts": 260,
    "Engagement Rate (%)": 4.1,
    "Avg Likes per Post": 213,
    "Avg Comments per Post": 9,
    "Verified": 0,
    "Account Age (Years)": 6.5,
    "Bio Text": "Teacher | Chennai | Book hoarder",
    "Account Type": "Real",
}


def frame(**overrides) -> pd.DataFrame:
    row = {**PROFILE, **overrides}
    return pd.DataFrame([row])


# ---- preprocessing ----------------------------------------------------------

def test_duplicate_rows_are_removed():
    df = pd.DataFrame([PROFILE, PROFILE, {**PROFILE, "Followers": 99}])
    out, report = clean(df)
    assert report.rows_in == 3
    assert report.dropped_duplicates == 1
    assert len(out) == 2


def test_rows_without_a_label_are_dropped_not_imputed():
    """Imputing the target would be inventing ground truth."""
    df = pd.DataFrame([PROFILE, {**PROFILE, "Account Type": None}])
    out, report = clean(df)
    assert report.dropped_missing_target == 1
    assert len(out) == 1


def test_missing_numeric_is_imputed_with_the_median_and_recorded():
    df = pd.DataFrame([
        {**PROFILE, "Followers": 100},
        {**PROFILE, "Followers": 300, "Posts": 1},
        {**PROFILE, "Followers": None, "Posts": 2},
    ])
    out, report = clean(df)
    assert report.imputed["followers"] == 1
    assert report.impute_values["followers"] == 200.0
    assert out["followers"].isna().sum() == 0
    # The median is recorded for every numeric column, not only the ones that
    # had a gap, because the API needs a fill value for omitted optional fields.
    assert "posts" in report.impute_values


def test_cleaning_normalises_either_dataset_shape():
    """The 15-column Instagram export and the generator output both load."""
    out, _ = clean(frame())
    assert {"full_name", "profile_picture", "external_url", "followers"} <= set(out.columns)


# ---- feature engineering ----------------------------------------------------

def test_table_1_profile_features_are_built():
    out = features.build_feature_frame(frame())
    row = out.iloc[0]
    assert row["profile_pic"] == 1
    assert row["len_fullname"] == len("Priya Kumar")
    assert row["fullname_words"] == 2
    assert row["ratio_numlen_fullname"] == 0.0
    assert row["len_desc"] == len("Teacher | Chennai | Book hoarder")
    assert row["external_url"] == 0
    assert row["posts"] == 260 and row["followers"] == 5200 and row["follows"] == 410


def test_digits_in_the_name_raise_the_numeric_ratio():
    """The paper's ratio_numlen_fullname: fakes pad names with digit runs."""
    plain = features.build_feature_frame(frame(**{"Real Name": "Priya Kumar"})).iloc[0]
    junky = features.build_feature_frame(frame(**{"Real Name": "priya88421"})).iloc[0]
    assert plain["ratio_numlen_fullname"] == 0.0
    assert junky["ratio_numlen_fullname"] == pytest.approx(5 / 10)


def test_behavioural_ratios_survive_a_zero_denominator():
    """A brand-new account has zero followers; a ratio must not become inf/NaN."""
    out = features.build_feature_frame(frame(**{
        "Followers": 0, "Avg Likes per Post": 0, "Account Age (Years)": 0,
    }))
    row = out.iloc[0]
    for column in ("follows_per_follower", "likes_per_follower", "comments_per_like"):
        assert pd.notna(row[column]) and row[column] != float("inf")
    assert row["posts_per_year"] > 0  # age is clipped, not divided by zero


def test_textual_features_flag_spam_shaped_bios():
    spam = features.build_feature_frame(frame(**{
        "Bio Text": "FREE FOLLOWERS >>> click link in bio DM for promo"})).iloc[0]
    normal = features.build_feature_frame(frame()).iloc[0]
    assert spam["bio_has_url"] == 1 and normal["bio_has_url"] == 0
    assert spam["bio_promo_terms"] > normal["bio_promo_terms"]
    assert spam["bio_upper_ratio"] > normal["bio_upper_ratio"]


def test_unavailable_paper_attributes_are_reported_not_invented():
    """No dataset here has a username or a private flag. The pipeline must say
    so rather than producing a column of zeros that looks like a measurement."""
    built = features.build_feature_frame(frame())
    for column in ("private", "ratio_numlen_username", "sim_name_username"):
        assert column not in built.columns

    avail = features.availability(set(features.normalise_columns(frame()).columns))
    assert set(avail["unavailable"]) == {
        "Private", "ratio_numlen_username", "sim_name_username"}
    assert "Profile Picture" in avail["available"]


def test_username_features_appear_when_the_column_does():
    """The code is present and correct; only the data is missing."""
    df = frame()
    df["Username"] = "priyakumar_7"
    built = features.build_feature_frame(df)
    assert built.iloc[0]["ratio_numlen_username"] == pytest.approx(1 / 12)
    assert built.iloc[0]["sim_name_username"] > 0


# ---- class imbalance --------------------------------------------------------

def test_strategy_is_only_applied_when_the_data_is_skewed():
    assert choose_strategy("auto", 1.0)[0] == "none"
    assert choose_strategy("auto", 0.98)[0] == "none"
    assert choose_strategy("auto", 1.73)[0] == "class-weight"
    assert choose_strategy("auto", 19.0)[0] == "class-weight"
    assert choose_strategy("oversample", 1.0)[0] == "oversample"


def test_imbalance_ratio_counts_negatives_per_positive():
    assert imbalance_ratio({"Real": 900, "Fake": 100}, "Fake") == 9.0


def test_oversampling_balances_and_only_grows():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(["Real"] * 90 + ["Fake"] * 10)
    Xo, yo = oversample(X, y, seed=1)
    assert len(Xo) == len(yo) == 180
    assert yo.value_counts().to_dict() == {"Real": 90, "Fake": 90}
    # Nothing was discarded: every original row is still represented.
    assert set(X["a"]) <= set(Xo["a"])


# ---- the join between training and serving ----------------------------------

@pytest.mark.skipif(
    not (BACKEND / "artifacts" / "xgb" / "model_meta.json").exists(),
    reason="XGBoost artifact not trained; run python -m ml.train_xgb",
)
def test_api_features_match_training_features():
    """The exact columns, in the exact order, that the model was fitted on.

    This is the test that would have caught the original project's central bug,
    where the bio was encoded one way during training and another way at predict
    time and every request 500'd.
    """
    from app.services.model_service import ModelService

    meta = json.loads((BACKEND / "artifacts" / "xgb" / "model_meta.json").read_text())
    service = ModelService(
        BACKEND / "artifacts" / "xgb" / "model.joblib",
        BACKEND / "artifacts" / "xgb" / "model_meta.json",
    )
    service.load()

    payload = {
        "followers": 5200, "following": 410, "posts": 260, "engagement_rate": 4.1,
        "avg_likes_per_post": 213, "avg_comments_per_post": 9, "verified": False,
        "account_age_years": 6.5, "bio_text": "Teacher | Chennai | Book hoarder",
        "full_name": "Priya Kumar", "profile_picture": True,
        "external_url": False, "language": "English",
    }
    built, imputed = service._to_engineered_frame([payload])

    assert list(built.columns) == meta["feature_columns"]
    assert imputed == [[]], "a complete payload must impute nothing"

    # And the same profile arriving as a CSV row through the training path must
    # produce identical values -- same function, same numbers.
    training_side = features.build_feature_frame(frame())[meta["feature_columns"]]
    pd.testing.assert_frame_equal(
        built.reset_index(drop=True).astype(str),
        training_side.reset_index(drop=True).astype(str),
    )


@pytest.mark.skipif(
    not (BACKEND / "artifacts" / "xgb" / "model_meta.json").exists(),
    reason="XGBoost artifact not trained; run python -m ml.train_xgb",
)
def test_omitted_optional_fields_are_imputed_and_named():
    from app.services.model_service import ModelService

    service = ModelService(
        BACKEND / "artifacts" / "xgb" / "model.joblib",
        BACKEND / "artifacts" / "xgb" / "model_meta.json",
    )
    service.load()
    _built, imputed = service._to_engineered_frame([{
        "followers": 5200, "following": 410, "posts": 260, "engagement_rate": 4.1,
        "avg_likes_per_post": 213, "avg_comments_per_post": 9, "verified": False,
        "account_age_years": 6.5, "bio_text": "Teacher | Chennai | Book hoarder",
    }])
    assert set(imputed[0]) >= {"full_name", "profile_picture", "external_url"}
