#!/usr/bin/env python3
"""Create a human-readable P111 scientific closure from the frozen JSON receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def hash_file(path: Path) -> str:
    """Compute a streaming source digest."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize(path: Path, target: Path) -> None:
    """Verify preregistered checks and write the scientific interpretation."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["experiment"] == "P111"
    assert receipt["status"] == "COMPLETE"
    assert len(receipt["per_image"]) == 180
    assert receipt["summary"]["changed_final_masks"] == 12
    assert receipt["summary"]["changed_final_images"] == 11
    assert all(receipt["synthetic_controls"].values())
    rows = receipt["per_image"]
    changed = [r for r in rows if r["different_final_masks"]]
    unchanged = [r for r in rows if not r["different_final_masks"]]
    assert len(changed) == 11 and len(unchanged) == 169

    x_in_changed = sum(int(r["crop_comparisons"]["X"]["different_elements"] > 0) for r in changed)
    x_in_unchanged = sum(int(r["crop_comparisons"]["X"]["different_elements"] > 0) for r in unchanged)
    x_max = max(r["crop_comparisons"]["X"]["max_absolute_difference"] for r in rows)
    total_x_diff = sum(r["crop_comparisons"]["X"]["different_elements"] for r in rows)
    total_x = sum(r["crop_comparisons"]["X"]["elements"] for r in rows)

    model_stats = {}
    for model in ("resnet34", "efficientnet_v2_s"):
        covered = [r["refiner_comparisons"][model] for r in rows if model in r["refiner_comparisons"]]
        assert len(covered) == 176
        model_stats[model] = {
            "images_exact": sum(int(r["exact"]) for r in covered),
            "images_changed": sum(int(not r["exact"]) for r in covered),
            "elements_changed": sum(r["different_elements"] for r in covered),
            "elements_total": sum(r["elements"] for r in covered),
            "maximum_absolute_difference": max(r["max_absolute_difference"] for r in covered),
            "threshold_0_5_crossings": sum(r["threshold_half_crossings_descriptive_only"] for r in covered),
        }

    metadata_exact = {name: sum(int(r["crop_comparisons"][name]["exact"]) for r in rows)
                      for name in ("M", "comp", "score")}
    assert all(value == 180 for value in metadata_exact.values()), metadata_exact
    assert all(r["crop_changed"] for r in changed)
    assert not receipt["summary"]["falsified"]
    pilot = receipt["summary"]["pilot_images"]
    domain_path = path.with_name("P111_VALUE_DOMAIN_AUDIT.json")
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    for data in domain["archives"].values():
        assert data["minimum"] == 0.0 and data["maximum"] == 255.0
        assert data["above_one"] > 0 and data["nan_or_inf"] == 0

    explanation = """The primary falsifiable hypothesis survived this test: all 11 images with one or
more changed final masks had different upstream crop pixel tensors. This is an
association, not identification of the causal operator. The test cannot attribute a
specific threshold crossing to any particular final mask pixel. The archived, quantized uint8 refiner outputs also show additional numerical
drift, which often has no downstream effect. Reusing four P103 pilots limits like-for-like refiner comparison to 176 images.
No test ground truth, official PQ calculation, performance gain, or model promotion
is supported by these data."""

    md = f"""# P111 scientific closure — isolating clean-replay numerical drift

**Completed on 13 November 2026 UTC.** This study was preregistered and
SHA-256 frozen before execution. It audits the existing P103/P106 predictions;
it does not perform any Kaggle scoring, labels-based evaluation, training, or
new submission.

## Integrity and conditions
- Preregistration SHA-256: `{receipt['prereg_sha256']}`.
- Audit code SHA-256: `{receipt['source_sha256']}`.
- Full receipt SHA-256: `{hash_file(path)}`.
- Frozen P103 CSV SHA-256: `{receipt['frozen_inputs_sha256']['p103_csv']}`.
- Frozen P106 CSV SHA-256: `{receipt['frozen_inputs_sha256']['p106_csv']}`.
- P107 independent per-image baseline verified: 180 images, 12 final
  mask differences in 11 images, with no matched-instance reassignment.
- Three synthetic controls passed, including the FP32 one-ULP half-threshold case.
- CPU only, CUDA invisible, no labels or generated submission. Original inputs
  remained read-only and outputs remained on the project HDD.

## Measured differences

| Quantity | Result |
| --- | ---: |
| Images with changed P103/P106 output | 11/180 |
| Final masks with one-pixel difference | 12/1,241 |
| Images with changed crop pixels X | {x_in_changed + x_in_unchanged}/180 |
| Output-changed images with different X | {x_in_changed}/11 |
| Output-unchanged images with different X | {x_in_unchanged}/169 |
| Different crop X values | {total_x_diff:,} / {total_x:,} compared positions |
| Largest absolute crop pixel drift | {x_max:.10g} |
| Images with exactly identical M, comp, score | {min(metadata_exact.values())}/180 |
| P103 pilot images lacking new refiner probability files | {len(pilot)} |
| Output-changed images among those pilots | {len(receipt['summary']['changed_final_images_among_pilots'])} |

The different-X count is the count of positions unequal under exact comparison;
unchanged masks often arise despite upstream numerical differences. The crop
tensor total across all images was {total_x:,} values; the number differing was
{total_x_diff:,} values.

| Quantized refiner archive (176 comparable images) | Exact images | Different uint8 values | Max absolute level drift | Raw zero/nonzero transitions* |
| --- | ---: | ---: | ---: | ---: |
| ResNet34 | {model_stats['resnet34']['images_exact']}/176 | {model_stats['resnet34']['elements_changed']:,}/{model_stats['resnet34']['elements_total']:,} | {model_stats['resnet34']['maximum_absolute_difference']:.10g} | {model_stats['resnet34']['threshold_0_5_crossings']:,} |
| EfficientNetV2-S | {model_stats['efficientnet_v2_s']['images_exact']}/176 | {model_stats['efficientnet_v2_s']['elements_changed']:,}/{model_stats['efficientnet_v2_s']['elements_total']:,} | {model_stats['efficientnet_v2_s']['maximum_absolute_difference']:.10g} | {model_stats['efficientnet_v2_s']['threshold_0_5_crossings']:,} |

*The predeclared archive name “probabilities” is misleading: an independent
[domain audit](P111_VALUE_DOMAIN_AUDIT.json) (SHA-256 `ce7430180aab0bb75245fda5bfddef69591dcc8d0db42d27dc6e62a0eac0ea2d`) confirms **uint8 values 0–255**, not floating-point probabilities in [0,1].
A numerical 0.5 crossing in uint8 means only zero versus nonzero, **not** the
actual segmentation threshold; it cannot be interpreted as PQ or confidence.
The maximum differences above are raw integer intensity levels. The P103 and
P106 refiner inference used FP32 computation, but the archived arrays are quantized.

The four P103 pilot image IDs omitted from the fresh refiner comparison are
`{', '.join(pilot)}`. None had a changed final mask.

## Predeclared hypothesis and scientific interpretation
{explanation}

The strict provenance gate established by P108 continues to block the proposed
1.15x additional-context evaluation. Selected Kaggle H9b submission 58112557
(public PQ 0.39) is unchanged. P106 remains a separately reproducible,
unscored CSV. No new PQ value is inferred.

No JEV or Goldfish consultation was performed. The independent controls were
cross-input SHA-256 integrity checks, frozen P107 per-image evidence, direct
NumPy comparisons and positive/negative synthetic tests.

## Recommended next step
Preserve the evidence and the already-confirmed organizer submission. Further
scientific work should only target a demonstrably new hypothesis with a frozen
protocol and valid TRAIN out-of-sample provenance, or audit final artifact
integrity before the 15 November 00:00 UTC freeze. Do not rerun known-negative
experiments or launch redundant CUDA work.
"""
    target.write_text(md, encoding="utf-8")
    print(json.dumps({
        "report": str(target),
        "report_sha256": hash_file(target),
        "receipt_sha256": hash_file(path),
        "crop_x_drift_in_changed": x_in_changed,
        "crop_x_drift_in_unchanged": x_in_unchanged,
        "total_crop_x_different": total_x_diff,
        "total_crop_x_elements": total_x,
        "model_stats": model_stats,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.receipt, args.report)
