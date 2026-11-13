# P98 — Exact source-based reconstruction of the official detection aggregate

**Completed:** 13 November 2026 (UTC)  
**Scientific status:** positive, independently source-based provenance audit; no new Kaggle submission

## Finding

The missing aggregation function was located in the archived original scripts:
`testinf/dump_official.py` calls `make_submission.consensus` on all four detector checkpoints, each with normal and horizontally reflected views. The consensus rescales unsupported proposals (factor 0.75), applies a minimum score threshold, and performs deterministic cross-view mask-IoU suppression (IoU threshold 0.5).

P97 had tested simple concatenation, which is **not** what the original implementation does. P98 froze and executed the original archived consensus on the independently reconstructed P88 CPU FP32 proposal lists.

- **Pilot:** 4/4 original test images matched exactly.
- **Full audit:** **180/180** original test images matched exactly.
- **Equality criterion:** per-image proposal count, strict proposal order, full score and bounding-box fields, mask shape/dtype, and every pixel value.
- **Execution:** CPU only, `CUDA_VISIBLE_DEVICES` empty, no test ground-truth labels and no leaderboard-derived parameter selection.
- **Service:** `solarfil-p98-full-20261113.service`, completed with `Result=success`, `ExecMainStatus=0`.
- **Original aggregate SHA-256:** `930bd517b0b51906c210c01bd0de50e0f86cdd7d43cdb47caf6bcd6625e4bb32`.
- **P98 full receipt SHA-256:** `08f90053883f22d082d54e2bca10014afa9339ba6d7cd4f85e7d5ccdc2f3b612`.

The complete receipt `docs/submission/P98_180_AUDIT_20261113.json` includes source hashes, historical input hashes and all 180 image-level exact comparisons. The immutable original cache was used only as a **comparison target**, never to generate the candidate.

## Interpretation and remaining limitations

This result repairs the specific P97 detector-aggregation gap. It **does not** in itself establish that the whole independently run pipeline regenerates the final official CSV from original JPEGs: P96 previously observed numerical differences when replacing historical H3 maps, despite stable component selection in 180/180 images. P95 proved exact original-JPEG-to-crop equality only *when supplied with historical intermediate H3 and detector artifacts*. P94 separately verified exact final replay against the CSV actually uploaded to Kaggle. Those audit scopes are not interchangeable.

To complete the public reproducibility requirement, provide the source-faithful consensus in the public inference code, verify a cold notebook run from original JPEGs and downloadable weights, and either demonstrate final CSV identity or disclose measured non-identity.

## How to reproduce this local provenance audit

With the official unlabelled JPEGs, the P88 verified CPU FP32 detector evidence, the original archived `make_submission.py`, and frozen historical aggregate available locally:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/p98_source_consensus_parity.py --self-test
CUDA_VISIBLE_DEVICES='' python scripts/p98_source_consensus_parity.py --limit 4
CUDA_VISIBLE_DEVICES='' python scripts/p98_source_consensus_parity.py --limit 180
```

The archived historical cache is never shipped with public code and is not needed for ordinary inference; it is only used for scientific provenance comparison. This audit script alone is **not** the stand-alone inference notebook required by the competition.

## Submission status

Last verified official Kaggle PQ before this audit remains **0.39** (submission **58112557**, accepted). No new PQ has been measured or uploaded by P98. The mandatory final Google Form still requires a confirmed submission receipt. Priority remains completing the independent notebook and closing the final submission before 15 November 2026 at 00:00 UTC.
