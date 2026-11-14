# P114 scientific closure — image-level foreground union and exclusivity

**Executed:** 14 November 2026 at 08:10:12 UTC. **Status:** `PASS_ALL_CROSSCHECKS`.

## Provenance and preregistration

P114 was frozen before execution to test the pixel-exclusivity guarantee that P113's independently validated per-instance COCO RLE areas did not establish. Frozen protocol SHA-256: `7270fea87aef85d3d453d592d3491c37e98ade3c72f7ba35da0e97d4d4adf328`. Executed source SHA-256: `4b22a2ffef30a132c7cabcb0cbb235a40054cf1162bcd1b827c1c6ecf6d85565`. Receipt SHA-256: `eb1dadba2b40c08e2aa14c4240e524d68300ae4cb0c74d68bdf90d0d144bd2b8`; 180-row per-image CSV SHA-256: `6e60a86ae12854411814037831be69aa7490bf2848d16367b0592861fda509ec`.

The complete inputs were the genuine H9b submission CSV (SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`) and the independent P106 reconstruction (SHA-256 `16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3`). Original JPEG headers provided native dimensions. No TEST labels were accessed.

## Exact method

A self-contained Python COCO-RLE decoder generated flattened foreground intervals for every predicted instance (column-major 2048×2048 indexing). Per-image interval coalescence yielded the number of distinct foreground pixels. A separately implemented union via `pycocotools.mask.merge` agreed **exactly in all 180 images for each CSV**. Overlap excess equals the sum of individual instance areas minus foreground-union area. A linear two-pointer interval intersection computed H9b-vs-P106 foreground-union symmetric difference image by image. This method is deterministic and used CPU only; wall time was 0.55 seconds versus the preregistered 240-second limit.

## Results

| Measure | Genuine H9b | P106 public reconstruction |
| --- | ---: | ---: |
| Images / instances | 180 / 1,241 | 180 / 1,241 |
| Sum of instance-mask areas (pixels) | 2,481,790 | 2,481,766 |
| Sum of foreground-union areas (pixels) | 2,481,790 | 2,481,766 |
| Images with overlapping predicted instances | **0 / 180** | **0 / 180** |
| Total overlap-multiplicity excess (pixels) | **0** | **0** |
| Union-area independent cross-check failures | **0** | **0** |

Across both predictions the foreground-union symmetric difference is **270 pixels** across **89 of 180 images**. The maximum per-image union difference is **22 pixels**; the net change in union area is **−24 pixels** (P106 minus H9b). The 270 union-difference pixels are consistent with the 270 summed matched-instance symmetric-difference pixels reported by P107. This corroborates a very small reconstruction variation and, when combined with P107's verified one-to-one matching, is consistent with no between-instance reassignment in the changed pixels. P114 itself audits foreground unions rather than independently recomputing P107's instance assignment.

## Scientific interpretation and decision

The hypothesis **that both output CSVs preserve one-instance-per-pixel exclusivity is supported over all 180 images**. In these two artifacts, summed instance area equals foreground union area; the −24 area difference reported in P113 is therefore also the net change of unique foreground pixels, **not** a PQ or accuracy difference. The 270 pixels are an image-prediction consistency statistic against a previous prediction, not ground-truth errors. No official PQ has been computed for P106.

P114 does not establish why differences occur or certify OOF provenance for the proposed 1.15× context ablation. The P108 gate remains blocking; do not rerun negative experiments. H9b remains the selected Kaggle submission **58112557**, with last officially confirmed **0.39 public PQ**. No GPU work, training, hidden labels, leaderboard probing, Kaggle upload, form changes or final-selection changes occurred. JEV/Goldfish were not consulted for this deterministic test.

Reproduce from `p114_union_exclusivity.py` and the frozen sources specified in `P114_PREREG.md`; inspect `P114_FINAL_RECEIPT.json` and the entire 180-image `P114_BY_IMAGE.csv`. This is new scientific release documentation only and does not alter the existing submitted four-page PDF, notebook, weights, CSV or organizer form.
