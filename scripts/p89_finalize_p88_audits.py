#!/usr/bin/env python3
"""Independently finalize P88 detector parity audits for all four temporal folds.

This tool reads unlabelled test JPEGs and existing CPU detector pickles. It
never starts CUDA, fits a model, accesses test labels, or submits to Kaggle.
"""
from __future__ import annotations

import argparse
import datetime as datetime
import json
import os
from pathlib import Path
import tempfile

from audit_p88_full_fold_parity import audit, selftest, sha

FOLDS = ("F1", "F2", "F3", "F4")
STAMP = "20261113"


def atomic_json(path: Path, data: dict) -> None:
    """Write a complete, sorted UTF-8 receipt with an atomic rename."""
    atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def atomic_text(path: Path, content: str) -> None:
    """Write and synchronously flush before replacing the destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".p88-audit-", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def verify_previously_published(documents: Path, result: dict) -> None:
    """Refuse to change the published F1/F2 evidence or silently accept drift."""
    fold = result["fold"]
    file = documents / f"P88_{fold}_INDEPENDENT_EXACT_RECEIPT_{STAMP}.json"
    old = json.loads(file.read_text(encoding="utf-8"))
    keys = (
        "fold", "images", "views", "exact_instances", "manifest_sha256",
        "reference_manifest_sha256", "source_sha256", "status",
    )
    differences = [key for key in keys if old.get(key) != result.get(key)]
    if differences:
        raise ValueError(f"{fold} does not agree with published receipt: {differences}")


def fold_markdown(result: dict) -> str:
    """Create an English, evidence-bounded report for one completed fold."""
    fold = result["fold"]
    return f"""# P88 — Independent exact CPU detector replay, {fold}

**Audit date (UTC):** 13 November 2026. **Status:** {result["status"]}.

The public Mask R-CNN FP32 {fold} weights and public implementation were
replayed from 180 original, unlabelled Kaggle test JPEG images on CPU.
The independent auditor compared each original JPEG SHA-256, every output
pickle SHA-256, the historical P69 reference, original checkpoint identity,
ordered confidence scores, bounding boxes and binary crop-mask bytes.
Both direct and horizontally mirrored detector views were checked.

| Verified measure | Result |
| --- | ---: |
| Original unlabelled images | {result["images"]}/180 |
| Detector views | {result["views"]}/360 |
| Exactly matching proposals | {result["exact_instances"]:,} |
| Divergent output image files | 0 |
| Test annotations accessed | No |
| New Kaggle submissions | No |

**Replay manifest SHA-256:** `{result["manifest_sha256"]}`

**P69 reference manifest SHA-256:** `{result["reference_manifest_sha256"]}`

**Auditor source SHA-256:** `{result["source_sha256"]}`

The executable independent check is
[`scripts/audit_p88_full_fold_parity.py`](../../scripts/audit_p88_full_fold_parity.py).
The accompanying JSON receipt is
[`P88_{fold}_INDEPENDENT_EXACT_RECEIPT_{STAMP}.json`](P88_{fold}_INDEPENDENT_EXACT_RECEIPT_{STAMP}.json).

**Strict scientific limitation:** this demonstrates detector-stage equality
with an archived local reference, not equality of final masks or of the
accepted H9b CSV, and not a new PQ measurement. Original competition test
labels were never used. The accepted Kaggle baseline remains submission
58112557 (public PQ 0.39); this audit is not evidence for promoting a model.
The end-to-end reproduction notebook and final submission form receipt
remain distinct, unfulfilled final-delivery gates.
"""


def summary_markdown(results: list[dict], summary: dict) -> str:
    """Report exact parity with clear scope, limitations and next steps."""
    lines = [
        "# P88 — Completed independent four-fold FP32 CPU detector parity",
        "",
        "**Audit date (UTC):** 13 November 2026. **Status:** four folds independently verified.",
        "",
        "The original, unlabelled 180-image test-domain inventory was processed",
        "by each of four released detectors on CPU, in two views per fold.",
        "Each output was independently compared with archived P69 detector",
        "outputs, including input and output SHA-256, ordered scores, boxes",
        "and per-instance binary masks. No new training or GPU inference was run.",
        "",
        "| Fold | Original images | Views | Exactly matching proposals |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f'| {result["fold"]} | {result["images"]} | '
            f'{result["views"]} | {result["exact_instances"]:,} |'
        )
    lines.extend([
        f'| **Total image-fold pairs** | **{summary["image_fold_pairs"]}** | '
        f'**{summary["detector_views"]}** | **{summary["exact_instances"]:,}** |',
        "",
        "**Important interpretation:** the test inventory consists of 180",
        "distinct unlabelled original images, each evaluated under four",
        "detector folds. Therefore, 720 is the number of image-fold pairs,",
        "not the number of different images.",
        "",
        "F1 and F2 receipts were compared with their already published immutable",
        "results; F3 and F4 were evaluated after the CPU replay completed.",
        "The independent auditor's self-test passed.",
        "",
        "**Scope:** detector-stage CPU parity only. This does not reproduce",
        "the full H9b stage-two selection, final-mask refinements or the",
        "accepted Kaggle CSV. PQ cannot be inferred from exact detector parity.",
        "The previously identified 325/1,241 RLE-row disagreement in the",
        "stage-two replay has not been resolved by this audit.",
        "",
        "**Competition:** retain baseline Kaggle submission 58112557,",
        "public score 0.39, without a replacement until independent validation.",
        "Do not use competition test annotations or hidden leaderboard feedback",
        "for model selection.",
        "",
        "**Unresolved mandatory gates:** exact original-image-to-final-CSV",
        "notebook and proof of the official final Google Form submission.",
        "Freeze all final deliverables by 15 November 2026 at 00:00 UTC.",
        "",
        f'**Machine-readable summary:** '
        f'[`P88_FOURFOLD_INDEPENDENT_AUDIT_RECEIPT_{STAMP}.json`]'
        f'(P88_FOURFOLD_INDEPENDENT_AUDIT_RECEIPT_{STAMP}.json).',
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    """Run read-only audits, reconcile published receipts and write evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    selftest()
    if args.self_test:
        print("P88_FOURFOLD_SELFTEST_OK")
        return
    results = [
        audit(args.root, args.reference, args.images, fold)
        for fold in FOLDS
    ]
    for result in results[:2]:
        verify_previously_published(args.documents, result)
    if len({r["fold"] for r in results}) != len(FOLDS):
        raise RuntimeError("Duplicate or missing temporal fold")
    if any(r["images"] != 180 or r["views"] != 360 for r in results):
        raise RuntimeError("Incomplete full-domain replay")
    total = sum(result["exact_instances"] for result in results)
    summary = dict(
        status="P88_FOURFOLD_INDEPENDENT_PARITY_COMPLETE",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        distinct_unlabelled_images=180,
        image_fold_pairs=sum(r["images"] for r in results),
        detector_views=sum(r["views"] for r in results),
        exact_instances=total,
        folds=results,
        auditor_sha256=sha(Path(__file__)),
        cpu_only=True,
        no_competition_test_labels=True,
        no_gpu_jobs_started=True,
        no_new_kaggle_submissions=True,
        end_to_end_csv_reproduced=False,
        final_google_form_receipt_verified=False,
    )
    for result in results[2:]:
        fold = result["fold"]
        atomic_json(
            args.documents / f"P88_{fold}_INDEPENDENT_EXACT_RECEIPT_{STAMP}.json",
            result,
        )
        atomic_text(
            args.documents / f"P88_{fold}_INDEPENDENT_EXACT_RESULT_{STAMP}.md",
            fold_markdown(result),
        )
    atomic_json(
        args.documents / f"P88_FOURFOLD_INDEPENDENT_AUDIT_RECEIPT_{STAMP}.json",
        summary,
    )
    atomic_text(
        args.documents / f"P88_FOURFOLD_INDEPENDENT_SUMMARY_{STAMP}.md",
        summary_markdown(results, summary),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
