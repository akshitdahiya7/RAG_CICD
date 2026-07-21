"""Download the NIST SP 800-53 Rev 5 OSCAL catalog to data/raw/.

Re-run this any time to refresh the local copy. The file isn't committed to
git (see .gitignore) — this script is the reproducible source of truth.
"""
import sys
from pathlib import Path

import requests

CATALOG_URL = (
    "https://raw.githubusercontent.com/usnistgov/oscal-content/main/"
    "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
)
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "nist_800_53_rev5_catalog.json"


def fetch() -> Path:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CATALOG_URL}")
    response = requests.get(CATALOG_URL, timeout=30)
    response.raise_for_status()
    DEST.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes -> {DEST}")
    return DEST


if __name__ == "__main__":
    try:
        fetch()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)





