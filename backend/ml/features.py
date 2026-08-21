#!/usr/bin/env python3
"""Feature engineering for the paper's fake-profile pipeline.

Paper: *Fake Profile Detection Using XGBoost Algorithm* (Venkadesh et al.,
ICRDICCT'25), Table 1 and section 4.2.

**This module is the single source of truth for what a feature means.** Training
(`ml.train_xgb`) and inference (`app.services.model_service`) both call
`build_feature_frame`, so a feature can never be computed one way during fitting
and another way at request time. That is the whole point: the original Flask app
label-encoded the bio during training and re-encoded it differently at predict
time, and every request 500'd.

What the paper asks for, and what this dataset can actually support:

| Paper attribute (Table 1)   | Engineered here            | Source column      |
|-----------------------------|----------------------------|--------------------|
| Profile Picture             | `profile_pic`              | `Profile Picture`  |
| Full name words             | `fullname_words`           | `Real Name`        |
| (len_fullname, sec 5.1)     | `len_fullname`             | `Real Name`        |
| (ratio_numlen_fullname)     | `ratio_numlen_fullname`    | `Real Name`        |
| Bio/Description length      | `len_desc`                 | `Bio Text`         |
| External URL                | `external_url`             | `Profile Link`     |
| Posts                       | `posts`                    | `Posts`            |
| Followers                   | `followers`                | `Followers`        |
| Follows                     | `follows`                  | `Following`        |
| **Private**                 | —                          | **not in dataset** |
| **ratio_numlen_username**   | —                          | **not in dataset** |
| **sim_name_username**       | —                          | **not in dataset** |

The three marked rows are **not fabricated**. No dataset in this repository
carries a username or a private-account flag, so those features are reported as
unavailable by `availability()` and recorded in the model metadata, rather than
being invented from a random draw.

Beyond Table 1 the paper's section 4.2 asks for *activity/behavioural* features
("activity logs, post frequency") and *textual* NLP features. Those are the
`behavioural` and `textual` groups below.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET = "Account Type"
POSITIVE_CLASS = "Fake"

# --------------------------------------------------------------- raw schema
# Canonical raw field -> the column headers that have carried it in this repo.
# Two dataset shapes exist (the 15-column Instagram export and the 10-column
# generator output); one alias table means neither needs a bespoke loader.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "full_name": ("Real Name", "Full Name", "full_name"),
    "bio_text": ("Bio Text", "bio_text", "description"),
    "profile_picture": ("Profile Picture", "profile_pic", "profile_picture"),
    "external_url": ("Profile Link", "External URL", "external_url"),
    "private": ("Private", "private"),
    "username": ("Username", "username"),
    "posts": ("Posts", "#posts", "posts"),
    "followers": ("Followers", "#followers", "followers"),
    "following": ("Following", "#follows", "following"),
    "verified": ("Verified", "verified"),
    "account_age_years": ("Account Age (Years)", "account_age_years"),
    "engagement_rate": ("Engagement Rate (%)", "engagement_rate"),
    "avg_likes_per_post": ("Avg Likes per Post", "avg_likes_per_post"),
    "avg_comments_per_post": ("Avg Comments per Post", "avg_comments_per_post"),
    "engagement_consistency": ("Engagement Consistency (%)", "engagement_consistency"),
    "language": ("Language", "language"),
}

# Paper attribute -> (engineered column, raw field it needs). A raw field that
# no dataset provides makes the attribute unavailable; see `availability()`.
PAPER_FEATURE_SOURCES: dict[str, tuple[str, str]] = {
    "Profile Picture":       ("profile_pic", "profile_picture"),
    "Full name words":       ("fullname_words", "full_name"),
    "len_fullname":          ("len_fullname", "full_name"),
    "ratio_numlen_fullname": ("ratio_numlen_fullname", "full_name"),
    "Bio/Description length": ("len_desc", "bio_text"),
    "External URL":          ("external_url", "external_url"),
    "Private":               ("private", "private"),
    "ratio_numlen_username": ("ratio_numlen_username", "username"),
    "sim_name_username":     ("sim_name_username", "username"),
    "Posts":                 ("posts", "posts"),
    "Followers":             ("followers", "followers"),
    "Follows":               ("follows", "following"),
}

TEXT_COLUMN = "bio_clean"
# The one genuinely categorical field in this data. The paper's preprocessing
# stage calls for "converting categorical variables to numerical formats
# (e.g., encoding)"; it is one-hot encoded *inside* the fitted pipeline with
# handle_unknown="ignore", so a language never seen in training cannot shift the
# column layout at request time.
CATEGORICAL_COLUMN = "language"

_DIGITS = re.compile(r"\d")
_WORD = re.compile(r"[^\s]+")
_URLISH = re.compile(r"(https?://|www\.|\.com|\.io|\.ly|link in bio)", re.IGNORECASE)
_PROMO = re.compile(
    r"\b(dm|promo|free|click|follow ?back|f4f|l4l|cheap|buy|earn|crypto|invest|win|"
    r"guarantee[d]?|cash|\$\$)\b",
    re.IGNORECASE,
)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever aliases a CSV happens to use to the canonical raw names.

    Unknown columns are left untouched; the target column is preserved as-is.
    """
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename[alias] = canonical
                break
    return df.rename(columns=rename)


def availability(raw_columns: set[str]) -> dict[str, list[str]]:
    """Which Table 1 attributes this data supports, and which it cannot.

    Called by the trainer so the answer is recorded in the artifact metadata
    instead of living only in a README.
    """
    available, unavailable = [], []
    for attribute, (_, needs) in PAPER_FEATURE_SOURCES.items():
        (available if needs in raw_columns else unavailable).append(attribute)
    return {"available": available, "unavailable": unavailable}


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio that yields 0 rather than inf/NaN on a zero divisor."""
    denom = denominator.astype(float)
    out = np.divide(
        numerator.astype(float), denom,
        out=np.zeros(len(numerator), dtype=float), where=denom > 0,
    )
    return pd.Series(out, index=numerator.index)


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw profile columns -> the model's feature frame.

    `raw` may use either the CSV headers or the canonical names; it is normalised
    first. Only features whose source column is present are produced, so the
    returned column list depends on the dataset — which is why the trainer records
    that list in the metadata and the API asserts against it at load time.
    """
    df = normalise_columns(raw).copy()
    out = pd.DataFrame(index=df.index)

    # -- Table 1: profile features ----------------------------------------
    if "profile_picture" in df:
        out["profile_pic"] = pd.to_numeric(df["profile_picture"], errors="coerce").fillna(0).astype(int)

    if "full_name" in df:
        name = _text(df["full_name"])
        out["len_fullname"] = name.str.len()
        out["fullname_words"] = name.apply(lambda s: len(_WORD.findall(s)))
        digits = name.apply(lambda s: len(_DIGITS.findall(s)))
        out["ratio_numlen_fullname"] = _ratio(digits, out["len_fullname"])

    if "bio_text" in df:
        bio = _text(df["bio_text"])
        out["len_desc"] = bio.str.len()

    if "external_url" in df:
        out["external_url"] = pd.to_numeric(df["external_url"], errors="coerce").fillna(0).astype(int)

    if "private" in df:  # not present in any dataset here; honoured if one ever is
        out["private"] = pd.to_numeric(df["private"], errors="coerce").fillna(0).astype(int)

    if "username" in df:  # ditto - the two username features arrive with it
        user = _text(df["username"])
        u_len = user.str.len()
        out["ratio_numlen_username"] = _ratio(user.apply(lambda s: len(_DIGITS.findall(s))), u_len)
        if "full_name" in df:
            out["sim_name_username"] = [
                _token_overlap(a, b) for a, b in zip(_text(df["full_name"]), user)
            ]

    for canonical, column in (("posts", "posts"), ("followers", "followers"), ("following", "follows")):
        if canonical in df:
            out[column] = pd.to_numeric(df[canonical], errors="coerce").fillna(0)

    # -- section 4.2: behavioural / activity features ----------------------
    if "verified" in df:
        out["verified"] = pd.to_numeric(df["verified"], errors="coerce").fillna(0).astype(int)
    if "account_age_years" in df:
        out["account_age_years"] = pd.to_numeric(df["account_age_years"], errors="coerce").fillna(0.0)
    if "engagement_rate" in df:
        out["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce").fillna(0.0)
    if "avg_likes_per_post" in df:
        out["avg_likes_per_post"] = pd.to_numeric(df["avg_likes_per_post"], errors="coerce").fillna(0.0)
    if "avg_comments_per_post" in df:
        out["avg_comments_per_post"] = pd.to_numeric(df["avg_comments_per_post"], errors="coerce").fillna(0.0)
    if "engagement_consistency" in df:
        out["engagement_consistency"] = pd.to_numeric(df["engagement_consistency"], errors="coerce").fillna(0.0)

    # Derived behaviour. These encode the actual mechanics of fake engagement:
    # mass-follow farming shows up in the follow ratio, bought likes show up as
    # likes that do not match the audience, and bots almost never comment.
    if {"followers", "follows"} <= set(out.columns):
        out["follows_per_follower"] = _ratio(out["follows"], out["followers"])
    if {"posts", "account_age_years"} <= set(out.columns):
        out["posts_per_year"] = _ratio(out["posts"], out["account_age_years"].clip(lower=0.05))
    if {"avg_likes_per_post", "followers"} <= set(out.columns):
        out["likes_per_follower"] = _ratio(out["avg_likes_per_post"], out["followers"])
    if {"avg_comments_per_post", "avg_likes_per_post"} <= set(out.columns):
        out["comments_per_like"] = _ratio(out["avg_comments_per_post"], out["avg_likes_per_post"])

    # -- section 4.2: textual / NLP features -------------------------------
    if "bio_text" in df:
        bio = _text(df["bio_text"])
        out["bio_is_empty"] = (bio.str.strip().isin(["", "(no bio)"])).astype(int)
        out["bio_word_count"] = bio.apply(lambda s: len(_WORD.findall(s)))
        out["bio_digit_ratio"] = _ratio(bio.apply(lambda s: len(_DIGITS.findall(s))), bio.str.len())
        out["bio_upper_ratio"] = _ratio(
            bio.apply(lambda s: sum(c.isupper() for c in s)),
            bio.apply(lambda s: sum(c.isalpha() for c in s)),
        )
        out["bio_has_url"] = bio.apply(lambda s: int(bool(_URLISH.search(s))))
        out["bio_promo_terms"] = bio.apply(lambda s: len(_PROMO.findall(s)))
        # Raw text for the TF-IDF branch of the pipeline. Kept as a string
        # column; the vectoriser is fitted inside the pipeline, never here.
        out[TEXT_COLUMN] = bio

    # -- categorical -------------------------------------------------------
    if "language" in df:
        out[CATEGORICAL_COLUMN] = _text(df["language"]).replace("", "unknown")

    return out


def _token_overlap(name: str, username: str) -> float:
    """Jaccard overlap of name tokens and username characters, 0..1."""
    a = {t.lower() for t in _WORD.findall(name)}
    b = username.lower()
    if not a or not b:
        return 0.0
    return sum(1 for token in a if token and token in b) / len(a)


def split_columns(frame: pd.DataFrame) -> tuple[list[str], str | None, str | None]:
    """(numeric columns, text column, categorical column) for a built frame.

    The trainer feeds these straight into the ColumnTransformer, so the three
    branches of the pipeline are derived from the data rather than hardcoded.
    """
    special = {TEXT_COLUMN, CATEGORICAL_COLUMN}
    numeric = [c for c in frame.columns if c not in special]
    text = TEXT_COLUMN if TEXT_COLUMN in frame.columns else None
    categorical = CATEGORICAL_COLUMN if CATEGORICAL_COLUMN in frame.columns else None
    return numeric, text, categorical
