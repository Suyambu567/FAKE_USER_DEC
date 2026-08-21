#!/usr/bin/env python3
"""Generate a realistic labelled dataset with actual signal.

This replaces `data/app.py`, which drew the label with
`np.random.choice(['Real','Fake'])` — independent of every feature. That is why
the shipped model scores 0.508 against a 0.508 baseline: there was nothing to
learn. Generating *more* rows from that script adds volume, not information.

Here the label comes first and the features are drawn from class-conditional
distributions grounded in how fake engagement actually behaves:

| signal                | real                          | fake                              |
|-----------------------|-------------------------------|-----------------------------------|
| following/follower    | low — people follow you back  | high — mass-follow to farm follows|
| engagement rate       | decays with audience size     | near-zero, or absurdly high (bought) |
| likes vs followers    | consistent                    | inconsistent with follower count  |
| comments per like     | ~1-5%                         | near zero (likes are botted, comments are not) |
| posts                 | accumulate over account age   | few, or bulk-dumped               |
| account age           | years                         | months                            |
| bio                   | personal, varied              | promo/spam keywords, or empty     |
| verified              | rare but real                 | almost never                      |
| profile picture       | almost always present         | missing on ~45%                   |
| full name             | two human tokens              | digits, keyword salad, single token |
| external URL          | uncommon                      | usual — the link is the point     |

Crucially the distributions **overlap**, and `--label-noise` flips a fraction of
labels outright. Without that, every algorithm hits 100% and a benchmark tells
you nothing. The defaults land the Bayes-optimal accuracy in the high 80s /
low 90s, which is the realistic band for this problem.

This is still synthetic. It is honest about being synthetic: it exists so the
pipeline, the API and the algorithm benchmark can be exercised against data that
*has* structure. **It is not a substitute for labelled real accounts** — a model
trained here learns these rules, not Instagram.

Usage:
    python -m ml.generate_dataset --rows 50000 --out ../data/synthetic_signal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REAL_BIOS = [
    "Passionate traveler and food lover", "Love exploring new places",
    "Photographer and content creator", "Entrepreneur & lifestyle blogger",
    "Digital nomad living my best life", "Sharing moments from my journey",
    "Personal blog | Stay positive", "Fitness enthusiast | Health advocate",
    "Nature lover | Saving the planet", "Aspiring artist | Drawing my dreams",
    "Foodie | Reviews and recipes", "Dad of two | Coffee first",
    "Mechanical engineer. Cyclist. Reader.", "Teacher | Chennai | Book hoarder",
    "Just here for the dog photos", "Chasing sunsets and good coffee",
    "Software developer | open source", "Mum | Baker | Garden in progress",
    "Music producer. DM for beats.", "Chartered accountant, amateur chef",
]

FAKE_BIOS = [
    "DM for promo!! Click link in bio", "FREE FOLLOWERS >>> link below",
    "Follow back 100% guaranteed", "CHECK MY LINK IN BIO FOR MORE",
    "Crypto investor | 500% returns | DM now", "Earn $$$ from home DM ME",
    "Buy followers cheap best price", "F4F L4L follow for follow",
    "Adult content click here", "WIN A FREE IPHONE CLICK NOW",
    "Investment guru DM for signals", "Get rich quick ask me how",
    "", "", "",  # empty bios are a strong fake signal
    "Promo promo promo DM", "Best deals link in bio click fast",
]


# --------------------------------------------------------------------------
# Profile fields for the paper's Table 1 (profile picture, full name, external
# URL). The 10-column output this generator used to produce had none of them,
# so the paper's profile-feature group could not be trained at all: the only
# dataset in the repo that *does* carry them (data/dataset.csv) has labels drawn
# by np.random.choice and therefore no signal to learn.
#
# The class-conditional choices below are the paper's own descriptions, not
# measurements: fake accounts more often have no profile picture, use
# alphanumeric or irregular names ("Phony profiles may utilize alphanumeric
# usernames or irregular naming conventions", section 5.1), and carry an
# external URL because the link is the point of the account. They are synthetic
# and the model metadata says so.
FIRST_NAMES = [
    "Isabella", "Evelyn", "Amelia", "Arjun", "Priya", "Daniel", "Mia", "Rahul",
    "Sofia", "Kenji", "Aisha", "Lucas", "Divya", "Noah", "Fatima", "Ravi",
    "Hannah", "Omar", "Ana", "Wei", "Leila", "Marcus", "Nithya", "Tomas",
]
LAST_NAMES = [
    "Jones", "Brown", "Davis", "Johnson", "Kumar", "Silva", "Nakamura", "Khan",
    "Garcia", "Okafor", "Rossi", "Iyer", "Novak", "Haddad", "Andersson", "Chen",
]
# Handle-shaped junk names, the "irregular naming conventions" the paper cites.
FAKE_NAME_STEMS = [
    "user", "official", "real", "the", "crypto", "promo", "insta", "shop",
    "deals", "girl", "boy", "vip", "free", "win",
]
LANGUAGES = ["English", "Spanish", "French", "German", "Japanese"]


def _real_names(rng, n):
    """Ordinary two-token human names; digits essentially never appear."""
    first = rng.choice(FIRST_NAMES, n)
    last = rng.choice(LAST_NAMES, n)
    names = np.char.add(np.char.add(first.astype(str), " "), last.astype(str))
    # A small minority use a single given name only.
    single = rng.random(n) < 0.12
    names[single] = first[single]
    return names


def _fake_names(rng, n):
    """Three shapes, mirroring how throwaway accounts are actually named."""
    stems = rng.choice(FAKE_NAME_STEMS, n).astype(str)
    firsts = rng.choice(FIRST_NAMES, n).astype(str)
    digits = rng.integers(0, 100000, n).astype(str)

    shape = rng.random(n)
    names = np.empty(n, dtype=object)
    for i in range(n):
        if shape[i] < 0.45:          # name + digit run -> high ratio_numlen_fullname
            names[i] = f"{firsts[i]}{digits[i]}"
        elif shape[i] < 0.70:        # generic single token -> short, one word
            names[i] = f"{stems[i]}{digits[i][:3]}"
        elif shape[i] < 0.85:        # padded keyword salad -> unusually long
            names[i] = f"{stems[i]} {stems[(i + 3) % n]} {firsts[i]} {digits[i]}"
        else:                        # a plain name; overlap is what makes this learnable
            names[i] = f"{firsts[i]} {rng.choice(LAST_NAMES)}"
    return names.astype(str)


def _lognormal(rng, mean_log, sigma, size, lo, hi):
    return np.clip(rng.lognormal(mean_log, sigma, size), lo, hi)


def generate(n: int, fake_ratio: float, label_noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fake = int(n * fake_ratio)
    n_real = n - n_fake
    labels = np.array(["Fake"] * n_fake + ["Real"] * n_real)

    # ---- real accounts ----------------------------------------------------
    r_followers = _lognormal(rng, 6.5, 1.6, n_real, 30, 5_000_000).astype(int)
    # People follow back, but the ratio falls as an account grows.
    r_ratio = rng.beta(2.2, 3.0, n_real) * (1.6 / (1 + np.log10(r_followers / 100 + 1)))
    r_following = np.clip(r_followers * r_ratio, 5, 7500).astype(int)
    # Engagement decays with audience size -- a real, well-documented effect.
    r_engagement = np.clip(
        rng.gamma(3.0, 1.4, n_real) / (1 + np.log10(r_followers / 500 + 1)), 0.2, 25.0
    )
    r_likes = np.clip(r_followers * r_engagement / 100 * rng.normal(1.0, 0.18, n_real), 0, None)
    # Real audiences comment at roughly 1-5% of the like rate.
    r_comments = np.clip(r_likes * rng.beta(2.0, 60.0, n_real), 0, None)
    r_age = np.clip(rng.gamma(3.2, 1.5, n_real), 0.2, 16)
    r_posts = np.clip(r_age * rng.gamma(2.5, 22, n_real), 1, 12_000).astype(int)
    r_verified = (rng.random(n_real) < np.clip(r_followers / 4_000_000, 0, 0.35)).astype(int)
    r_bio = rng.choice(REAL_BIOS, n_real)

    # ---- fake accounts ----------------------------------------------------
    f_followers = _lognormal(rng, 5.2, 1.5, n_fake, 10, 400_000).astype(int)
    # Mass-follow farming: following often exceeds followers.
    f_following = np.clip(
        f_followers * rng.gamma(3.0, 1.1, n_fake) + rng.integers(50, 2500, n_fake),
        20, 7500,
    ).astype(int)

    # Two fake sub-populations, which is why a linear boundary struggles here:
    #   ~65% dead accounts  -> almost no engagement
    #   ~35% bought engagement -> implausibly high for the follower count
    bought = rng.random(n_fake) < 0.35
    f_engagement = np.where(
        bought,
        np.clip(rng.gamma(9.0, 2.2, n_fake), 8, 70),
        np.clip(rng.gamma(1.1, 0.45, n_fake), 0.0, 4.0),
    )
    # Likes are decoupled from the real audience: heavy multiplicative noise.
    f_likes = np.clip(
        f_followers * f_engagement / 100 * rng.lognormal(0, 0.85, n_fake), 0, None
    )
    # Bots like; bots rarely comment. This ratio is the single strongest signal.
    f_comments = np.clip(f_likes * rng.beta(1.0, 400.0, n_fake), 0, None)
    f_age = np.clip(rng.gamma(1.2, 0.75, n_fake), 0.02, 6)
    f_posts = np.clip(rng.gamma(1.1, 9, n_fake), 0, 900).astype(int)
    f_verified = (rng.random(n_fake) < 0.004).astype(int)
    f_bio = rng.choice(FAKE_BIOS, n_fake)

    # ---- Table 1 profile fields ------------------------------------------
    # Most real accounts have a picture; a large minority of fakes do not.
    r_pic = (rng.random(n_real) < 0.96).astype(int)
    f_pic = (rng.random(n_fake) < 0.55).astype(int)
    # The external link is the reason a spam account exists; real accounts
    # mostly have nothing to link to.
    r_url = (rng.random(n_real) < 0.22).astype(int)
    f_url = (rng.random(n_fake) < 0.68).astype(int)
    r_name = _real_names(rng, n_real)
    f_name = _fake_names(rng, n_fake)
    # Language is drawn from the SAME distribution for both classes on purpose:
    # it exercises the categorical-encoding branch of the pipeline without
    # smuggling in signal that a real platform's language field would not carry.
    r_lang = rng.choice(LANGUAGES, n_real)
    f_lang = rng.choice(LANGUAGES, n_fake)

    df = pd.DataFrame({
        "Real Name": np.concatenate([f_name, r_name]),
        "Profile Picture": np.concatenate([f_pic, r_pic]),
        "Profile Link": np.concatenate([f_url, r_url]),
        "Language": np.concatenate([f_lang, r_lang]),
        "Followers": np.concatenate([f_followers, r_followers]),
        "Following": np.concatenate([f_following, r_following]),
        "Posts": np.concatenate([f_posts, r_posts]),
        "Engagement Rate (%)": np.round(np.concatenate([f_engagement, r_engagement]), 2),
        "Avg Likes per Post": np.concatenate([f_likes, r_likes]).astype(int),
        "Avg Comments per Post": np.concatenate([f_comments, r_comments]).astype(int),
        "Verified": np.concatenate([f_verified, r_verified]),
        "Account Age (Years)": np.round(np.concatenate([f_age, r_age]), 1),
        "Bio Text": np.concatenate([f_bio, r_bio]),
        "Account Type": labels,
    })

    # Label noise: real-world labelling is imperfect, and without this the task is
    # separable enough that every model saturates and the comparison is useless.
    if label_noise > 0:
        flip = rng.random(len(df)) < label_noise
        df.loc[flip, "Account Type"] = df.loc[flip, "Account Type"].map(
            {"Fake": "Real", "Real": "Fake"}
        )

    # A blank bio is itself a signal, but the API schema requires non-empty text.
    df["Bio Text"] = df["Bio Text"].replace("", "(no bio)")

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--fake-ratio", type=float, default=0.35,
                    help="Fraction of fake accounts. Real platforms sit at 5-10%%; "
                         "0.35 keeps the benchmark from being dominated by imbalance.")
    ap.add_argument("--label-noise", type=float, default=0.06,
                    help="Fraction of labels flipped. Caps the achievable accuracy, "
                         "which is what makes the algorithm comparison informative.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = generate(args.rows, args.fake_ratio, args.label_noise, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"wrote {args.out}  rows={len(df):,}")
    print(f"  class balance: {df['Account Type'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  distinct bios: {df['Bio Text'].nunique()}")
    print("\n  standardised mean difference by feature (>0.1 means real signal):")
    for col in ["Followers", "Following", "Posts", "Engagement Rate (%)",
                "Avg Likes per Post", "Avg Comments per Post", "Verified",
                "Account Age (Years)", "Profile Picture", "Profile Link"]:
        means = df.groupby("Account Type")[col].mean()
        smd = abs(means.iloc[0] - means.iloc[1]) / (df[col].std() or 1)
        print(f"    {col:26s} {smd:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
