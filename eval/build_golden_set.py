"""Build eval/golden_dataset.json — the reference Q&A set RAGAS grades against.

Two kinds of rows:
  - "auto"   — mechanically generated from a control's own statement text.
               Fully sourced from the catalog, no judgment involved.
  - "manual" — empty scaffold rows (including a few marked unanswerable) for
               you to fill in by hand, since writing a good question and
               judging what the catalog does *not* cover requires real
               judgment a template can't fake.

Run from the repo root:  python -m eval.build_golden_set
"""
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ingest.catalog_parser import CatalogParser
from ingest.models import Control

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "nist_800_53_rev5_catalog.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

PLACEHOLDER_RE = re.compile(r"\{\{\s*insert:\s*param,\s*[\w.-]+\s*\}\}")


@dataclass
class QAPair:
    id: str
    question: str
    ground_truth: str
    control_id: str
    source: str  # "auto" | "manual"
    answerable: bool = True


def clean_placeholders(text: str) -> str:
    """Replace unresolved OSCAL param placeholders (e.g. '{{ insert: param, ac-1_prm_1 }}')
    with a readable stand-in, since we're deliberately not resolving them to real values yet."""
    return PLACEHOLDER_RE.sub("[organization-defined parameter]", text)


def auto_seed_pairs(controls: list[Control], n: int, seed: int) -> list[QAPair]:
    """Mechanically build one Q&A pair per sampled base control from its statement."""
    base_controls = [c for c in controls if "." not in c.id]  # exclude enhancements like ac-2.1
    rng = random.Random(seed)
    sample = rng.sample(base_controls, min(n, len(base_controls)))

    pairs = []
    for control in sample:
        question = f"What does {control.id.upper()} ({control.title}) require?"
        ground_truth = clean_placeholders(control.statement)
        pairs.append(
            QAPair(
                id=f"auto-{control.id}",
                question=question,
                ground_truth=ground_truth,
                control_id=control.id,
                source="auto",
            )
        )
    return pairs


def manual_scaffold(n_answerable: int, n_unanswerable: int) -> list[QAPair]:
    """Empty rows to hand-write yourself: real questions phrased the way a person
    would actually ask, plus a few genuinely out-of-scope ones the assistant
    should recognize it can't answer rather than hallucinate."""
    pairs = []
    for i in range(1, n_answerable + 1):
        pairs.append(
            QAPair(id=f"manual-{i}", question="", ground_truth="", control_id="", source="manual")
        )
    for i in range(1, n_unanswerable + 1):
        pairs.append(
            QAPair(
                id=f"manual-unanswerable-{i}",
                question="",
                ground_truth="Not covered by the NIST 800-53 catalog.",
                control_id="",
                source="manual",
                answerable=False,
            )
        )
    return pairs


def build_golden_set(
    auto_n: int = 15, manual_n: int = 10, unanswerable_n: int = 3, seed: int = 42
) -> list[QAPair]:
    controls = CatalogParser().parse(CATALOG_PATH)
    return auto_seed_pairs(controls, auto_n, seed) + manual_scaffold(manual_n, unanswerable_n)


def main() -> None:
    pairs = build_golden_set()
    OUTPUT_PATH.write_text(
        json.dumps([asdict(p) for p in pairs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    auto_count = sum(p.source == "auto" for p in pairs)
    manual_count = sum(p.source == "manual" for p in pairs)
    print(f"Wrote {len(pairs)} Q&A pairs ({auto_count} auto, {manual_count} manual scaffold) -> {OUTPUT_PATH}")
    print("Fill in the empty 'manual' rows by hand before using this for eval.")


if __name__ == "__main__":
    main()
