"""Compose the hosted demo page: inline the exported data into the template.

The demo is served from GitHub Pages with no backend, so the payload is
embedded directly in the HTML rather than fetched. That keeps the page a
single self-contained file which works over file://, any static host, and
GitHub Pages identically — no CORS, no network dependency, nothing to break
while someone is looking at it.

Usage:  PYTHONPATH=. python scripts/build_demo.py
"""
from src.pipeline import config

TEMPLATE = config.PROJECT_ROOT / "frontend" / "demo_template.html"
DATA = config.PROJECT_ROOT / "docs" / "demo_data.json"
OUT = config.PROJECT_ROOT / "docs" / "index.html"
PLACEHOLDER = "__DEMO_DATA__"


def main():
    if not DATA.exists():
        raise SystemExit(
            f"{DATA} not found — run `python scripts/export_demo_data.py` first."
        )
    html = TEMPLATE.read_text()
    if PLACEHOLDER not in html:
        raise SystemExit(f"{PLACEHOLDER} placeholder missing from {TEMPLATE}")

    data = DATA.read_text()
    # Guard against the payload prematurely closing the inline <script> block.
    if "</script" in data.lower():
        raise SystemExit("Payload contains a closing script tag; refusing to inline.")

    OUT.write_text(html.replace(PLACEHOLDER, data))
    print(f"Built {OUT} ({OUT.stat().st_size / 1024:.0f} KB, data {len(data) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
