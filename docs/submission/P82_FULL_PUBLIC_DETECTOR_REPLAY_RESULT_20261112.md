# P82 — Full public Mask R-CNN original-image replay: completed negative byte-parity audit

**Executed 12 November 2026 UTC.** The frozen [P82 preregistration](P82_FULL_PUBLIC_DETECTOR_REPLAY_PREREG_20261112.md) required four public FP16 Mask R-CNN models × 180 distinct original unlabelled official Kaggle test JPEGs × original/reflected views, using no test labels or Kaggle API. All four public models loaded strictly after the documented P81 repair to filseg.common; source and checkpoint SHA-256 were verified. All 720 (model,image) combinations, hence 1,440 detector views, generated geometric-valid proposal crops. Every one of the 720 outputs has an immutable per-image pickle hash; each fold has a completed manifest and one aggregate P82 receipt. No regeneration of successful instances or concurrent CUDA processes was necessary.

| Public model | Original images | Both-view proposals | Same-index exact boxes vs historical FP32 | Same-index exact crop bytes |
| --- | ---: | ---: | ---: | ---: |
| F1 | 180 | 13,257 | 11,810 | 1 |
| F2 | 180 | 12,987 | 12,032 | 0 |
| F3 | 180 | 12,315 | 11,083 | 0 |
| F4 | 180 | 12,210 | 11,226 | 2 |
| **Total** | **720 image-model pairs** | **50,769** | **46,151** | **3** |

This is a **descriptive indexed-proposal comparison**, not instance-matched IoU or an accuracy metric. Discrepancies include changes from FP16 checkpoint rounding compared with the original FP32 CPU detector states; no conclusion is made about downstream PQ. Identical counts and 46,151 identical boxes do **not** imply byte-identical final masks. The F1–F4 source-to-detector step now runs with public code, but a byte-exact H9b final CSV is **not** established. All negative differences are retained.

**Execution evidence:** P82 began at 18:17:04 UTC and finished at 18:20:37 UTC, aggregate detector processing 208.84 s. One RTX 3090 GPU0 was used under the shared supervisor lock and physically identified via nvidia-smi, while GPU1 remained unused. The full set occupied 159 MiB of the project HDD; no large data were written to /mnt/datos. On the HDD at paired-small-20261112/p82_full_public_detector_20261112/ are immutable receipt.json, four F1–F4 manifest.json files with image/proposal hashes, and the append-only events.jsonl and p82_run.log. Public [replay script](../../scripts/p82_full_public_detector_replay.py) requires only the released weights, source modules and user-provided Kaggle pixels; the private P69 FP32 proposal reference is optional and must be a trusted locally produced pickle source. All model and image files are excluded from the Git repository to respect dataset terms.

### Reproducibility limits and final deliverables

The two public model-stage reconstructions are now: all 720 H3 U-Net semantic maps (P77–P80) and all 720 detector image/model outputs (P82). A **complete** original-image-to-final-CSV notebook still needs to recompute H6 candidate selection from openly reconstructable TRAIN-derived features, both crop refiners, fallback decisions and the final RLE assembly, followed by comparison against the **actually accepted** H9b CSV. The existing stage-two audit begins from cached intermediate predictions and its CSV digest differs from accepted submission **58112557**; neither model-stage reconstruction alone closes this gap.

The [four-page English report](TECHNICAL_REPORT_4P_20261112.pdf) is published. The [official mandatory final Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform) does **not** yet have a verified submitted-response receipt. The confirmed best official public PQ remains **0.39** (submission 58112557, 8 November 2026), unchanged by these reproducibility studies. No hidden test annotations or leaderboard-driven tuning was used.
