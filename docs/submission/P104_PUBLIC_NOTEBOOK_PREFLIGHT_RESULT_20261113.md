# P104 — Static and full-checkpoint preflight for the public H9b notebook

**Executed:** 13 November 2026, 10:40 UTC. **Scope:** non-CUDA notebook preflight; **status:** PASSED.

## Objective and reproducibility contract

The public clean raw-JPEG notebook at `notebooks/h9b_clean_raw_jpeg_to_csv.ipynb` must be independently inspectable and must not silently rely on historical Kaggle prediction caches or test annotations. Before any further GPU activity or optional new Kaggle submission, run a lightweight test of the released source interfaces and model integrity.

## Observed results

| Check | Result |
|---|---:|
| Notebook code cells compiled by CPython | 8/8 |
| Required public files found | 11/11 |
| Public function interfaces present | 4/4 |
| Checkpoint files present | 10/10 |
| Checkpoint SHA-256 values matching frozen manifest | 10/10 |
| Runtime packages discoverable (cv2, numpy, torch, pycocotools, PIL) | 5/5 |
| CUDA kernels or model inference executed by preflight | 0 |
| Kaggle submissions made | 0 |

Notebook SHA-256: `01a05524f2ca00e028d5e699ae38418eeba01f331db14246527bb4f87aba654c`.

The implementation and machine-readable receipt are in `scripts/preflight_h9b_clean_notebook.py` and `docs/submission/P104_PUBLIC_NOTEBOOK_FULL_HASH_RECEIPT_20261113.json`. The result is reproducible using:

```bash
python scripts/preflight_h9b_clean_notebook.py --root . --full-hash
```

## Interpretation and remaining work

This test establishes **static source compatibility and byte-integrity of the ten public weights**, not operational end-to-end replication. It does not execute H3 or Mask R-CNN on the 180 raw JPEGs, run the two refiners, calculate any PQ, or demonstrate equality against the historically accepted CSV.

The notebook explicitly uses a public deterministic no-component fallback instead of the earlier historical control predictions. It therefore must not be attributed the earlier official **PQ 0.39, Kaggle submission 58112557**, without a separate evaluated submission. The P103 full-domain coupling study is a separate preregistered CPU experiment and should finish before any scientifically justified follow-up.

The four-page PDF, a complete clean-run notebook validation, and an **actual receipt from the mandatory organiser Google Form** remain independent delivery requirements. Static preflight success is never a substitute for the form or a complete notebook execution.

No official TEST annotations, leaderboard-derived parameter changes, or competitor process modifications occurred during this test.
