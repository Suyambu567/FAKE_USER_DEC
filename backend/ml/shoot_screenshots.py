#!/usr/bin/env python3
"""Capture README screenshots from the *running* system.

Every image this writes is a real render of a real page talking to a real
server. Nothing is mocked, staged or drawn. If a page cannot be reached the
script fails loudly rather than producing a placeholder — a screenshot that
lies about what the project does is worse than no screenshot.

Prerequisites (the script checks each and tells you which is missing):

    # API
    cd backend && .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8677
    # static webapp
    cd webapp && python3 -m http.server 8451 --bind 127.0.0.1
    # figures
    cd backend && .venv/bin/python -m ml.visualize --dataset ../data/paper_signal.csv \
        --benchmark ../docs/benchmark_paper.json --out ../docs/figures

Run with the interpreter that has playwright installed (the system python here,
not backend/.venv):

    python3 -m ml.shoot_screenshots --out ../docs/screenshots
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8677"
WEBAPP = "http://127.0.0.1:8451"

# The profile every prediction screenshot uses. Spam-shaped on purpose: a
# screenshot of the system saying "Real" about an ordinary account demonstrates
# far less than one of it catching the thing it exists to catch.
FAKE_PROFILE = {
    "followers": 820,
    "following": 6900,
    "posts": 4,
    "engagement_rate": 0.3,
    "avg_likes_per_post": 2,
    "avg_comments_per_post": 0,
    "verified": False,
    "account_age_years": 0.2,
    "bio_text": "FREE FOLLOWERS >>> click link in bio DM for promo",
    "full_name": "crypto8842",
    "profile_picture": False,
    "external_url": True,
    "language": "English",
}


def open_operation(page, op_id: str):
    """Expand one Swagger operation and return a locator scoped to it.

    Deliberately not driven by `/docs#/tag/operationId`: Swagger UI is a single
    page, so navigating to a second hash does not reload and the expanded state
    from the previous operation persists — which made a header click *collapse*
    the block instead of opening it. Checking for the body and scoping every
    later selector to this block removes both ambiguities.
    """
    op = page.locator(f"#{op_id}")
    op.wait_for(state="visible", timeout=20_000)
    if op.locator(".opblock-body").count() == 0:
        op.locator(".opblock-summary").first.click()
        page.wait_for_selector(f"#{op_id} .opblock-body", timeout=20_000)
    op.scroll_into_view_if_needed()
    return op


def reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def shoot(out: Path, skip_webapp: bool, skip_figures: bool) -> int:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    figures = Path(__file__).resolve().parents[2] / "docs" / "figures"
    written: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 950},
                                device_scale_factor=2)

        # ---- 1. Swagger UI ------------------------------------------------
        page.goto(f"{API}/docs", wait_until="networkidle")
        page.wait_for_selector("#operations-tag-prediction", timeout=20_000)
        # Collapse the schema list so the endpoint table is what the reader sees.
        page.evaluate("""() => {
            const s = document.querySelector('section.models');
            if (s) s.style.display = 'none';
        }""")
        path = out / "swagger.png"
        page.screenshot(path=str(path), full_page=True)
        written.append(path)

        # ---- 2. POST /predict, Try it out --------------------------------
        page.goto(f"{API}/docs", wait_until="networkidle")
        predict_id = "operations-prediction-predict_one_api_v1_predict_post"
        op = open_operation(page, predict_id)
        op.locator("button.try-out__btn").click()
        op.locator("textarea.body-param__text").wait_for(timeout=20_000)
        op.locator("textarea.body-param__text").fill(json.dumps(FAKE_PROFILE, indent=2))
        page.wait_for_timeout(400)
        path = out / "prediction_request.png"
        op.screenshot(path=str(path))
        written.append(path)

        # ---- 3. Execute, and capture the real response --------------------
        op.locator("button.execute").click()
        # The response body only exists once the server has answered.
        op.locator(".responses-table.live-responses-table").wait_for(timeout=30_000)
        page.wait_for_timeout(800)
        body = op.locator(".live-responses-table .microlight").first.inner_text()
        if '"label"' not in body:
            print(f"error: /predict did not return a prediction:\n{body[:400]}",
                  file=sys.stderr)
            return 1
        print("  live prediction:", " ".join(body.split())[:160])
        path = out / "prediction_response.png"
        op.screenshot(path=str(path))
        written.append(path)

        # ---- 4. GET /model/info, the model's own evaluation ---------------
        # Fresh load so the expanded /predict block does not stretch the page.
        page.goto(f"{API}/docs", wait_until="networkidle")
        info_id = "operations-analytics-model_info_api_v1_model_info_get"
        op = open_operation(page, info_id)
        op.locator("button.try-out__btn").click()
        op.locator("button.execute").click()
        op.locator(".responses-table.live-responses-table").wait_for(timeout=30_000)
        page.wait_for_timeout(800)
        info = op.locator(".live-responses-table .microlight").first.inner_text()
        if '"accuracy"' not in info:
            print(f"error: /model/info returned no metrics:\n{info[:400]}", file=sys.stderr)
            return 1
        print("  live model info:", " ".join(info.split())[:160])
        path = out / "model_info.png"
        op.screenshot(path=str(path))
        written.append(path)

        # ---- 5. The generated figures -------------------------------------
        if not skip_figures:
            for svg, name in (("correlation_heatmap.svg", "model_evaluation_heatmap.png"),
                              ("algorithm_comparison.svg", "model_evaluation_algorithms.png")):
                source = figures / svg
                if not source.exists():
                    print(f"  skipping {name}: {source} not generated yet")
                    continue
                page.goto(source.as_uri(), wait_until="networkidle")
                box = page.locator("svg").bounding_box()
                page.set_viewport_size({"width": int(box["width"]) + 40,
                                        "height": int(box["height"]) + 40})
                page.wait_for_timeout(300)
                path = out / name
                page.locator("svg").screenshot(path=str(path))
                written.append(path)
            page.set_viewport_size({"width": 1440, "height": 950})

        # ---- 6. The static webapp ------------------------------------------
        if not skip_webapp:
            page.goto(WEBAPP, wait_until="networkidle")
            page.wait_for_timeout(1200)
            # Drive a real detection so the screenshot shows a verdict, not an
            # empty form. Field ids come from webapp/index.html.
            filled = page.evaluate("""(profile) => {
                const map = {
                  userId: profile.full_name,
                  followers: profile.followers, following: profile.following,
                  posts: profile.posts, engagement: profile.engagement_rate,
                  likes: profile.avg_likes_per_post, comments: profile.avg_comments_per_post,
                  age: profile.account_age_years, bio: profile.bio_text,
                };
                let n = 0;
                for (const [id, value] of Object.entries(map)) {
                  const el = document.getElementById(id);
                  if (!el) continue;
                  el.value = value;
                  el.dispatchEvent(new Event('input', {bubbles: true}));
                  el.dispatchEvent(new Event('change', {bubbles: true}));
                  n++;
                }
                const v = document.getElementById('verified');
                if (v && v.type === 'checkbox') {
                  v.checked = Boolean(profile.verified);
                  v.dispatchEvent(new Event('change', {bubbles: true}));
                }
                return n;
            }""", FAKE_PROFILE)
            print(f"  webapp: filled {filled} field(s)")
            page.locator("#analyzeBtn").click()
            page.wait_for_timeout(1800)
            verdict = page.locator("#verdictLabel").inner_text().strip()
            if not verdict:
                print("error: the webapp produced no verdict", file=sys.stderr)
                return 1
            print(f"  webapp verdict: {verdict}")
            # The page has a sticky header. In a full-page capture it renders at
            # whatever scroll offset the page is sitting at, landing a white bar
            # across the middle of the image and covering the verdict badge.
            # Scrolling home first puts it back where a reader would see it.
            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(600)
            path = out / "frontend.png"
            page.screenshot(path=str(path), full_page=True)
            written.append(path)

        browser.close()

    print("\nwrote:")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("../docs/screenshots"))
    ap.add_argument("--skip-webapp", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    if not reachable(f"{API}/health/live"):
        print(f"error: no API at {API}. Start it first:\n"
              "  cd backend && .venv/bin/python -m uvicorn app.main:app "
              "--host 127.0.0.1 --port 8677", file=sys.stderr)
        return 1
    if not args.skip_webapp and not reachable(WEBAPP):
        print(f"note: no webapp at {WEBAPP}; skipping frontend.png")
        args.skip_webapp = True

    return shoot(args.out.resolve(), args.skip_webapp, args.skip_figures)


if __name__ == "__main__":
    raise SystemExit(main())
