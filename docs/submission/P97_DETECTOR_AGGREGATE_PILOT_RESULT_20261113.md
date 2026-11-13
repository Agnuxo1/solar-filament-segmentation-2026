# P97 — Negative result: raw detector view concatenation does not reproduce the final official detection cache

**Date:** 13 November 2026 UTC  
**Design:** Preregistered four-image pilot using only unlabelled competition test images and previously independently verified model predictions.  
**Scientific status:** Completed pilot; two fixed natural concatenation orders rejected. No additional models were trained or run.

## Experimental question

P88 verified all 720 original per-fold Mask R-CNN source-model predictions with exact tuples and mask bytes. However, the historical no-site H6 crop stage consumes a **much smaller consolidated official detector cache**. P97 tested whether that aggregate is simply the unmodified eight prediction-view lists concatenated in either model-first or view-first order.

## Verified results

| Image (sorted pilot) | Raw proposals from F1–F4, two views | Historical consolidated proposals |
|---|---:|---:|
| 20110120105534Ch | 122 | 4 |
| 20110130110334Ch | 77 | 4 |
| 20110214175414Mh | 188 | 11 |
| 20110329082654Uh | 395 | 29 |

Neither frozen order matched for any of the four images: **0/4 exact** for model-first, **0/4 exact** for view-first. The counts alone rule out raw concatenation. It is therefore necessary to identify and document the actual upstream postprocessing/merging/filtering operation; the precise mechanism is not established by this experiment.

Both the test harness and deliberate negative cases passed. The historical aggregate was read only. The complete P97 JSON receipt includes all count and strict comparison results.

## Consequences for deliverables

The public FP32 weight reconstruction is genuine and independently demonstrated, but the historical aggregate is **not automatically reconstructable by concatenating the per-fold arrays**. Do not advertise original-JPEG-to-final-CSV exact reproduction on this basis.

The separate P95 work proved original-JPEG-to-H6-crops parity *using historical H3 and consolidated detector outputs*. P96 is currently investigating replacing H3 historical caches with publicly replayed H3 maps. The highest known Kaggle public PQ remains 0.39 from submission 58112557; no new submission, leaderboard analysis or test-label access was involved.

## Reproduction

Run `CUDA_VISIBLE_DEVICES='' python scripts/p97_public_detector_aggregation.py --self-test` and then `CUDA_VISIBLE_DEVICES='' python scripts/p97_public_detector_aggregation.py --limit 4` from the repository checkout, with the P88 evidence directory and frozen original aggregate present. The original aggregate is never publicly bundled with the code; the audit is an internal provenance check, not a clean-install notebook.

**Next research gate:** locate the archived `dump_official.py` (or equivalent) and preregister an exact source-based aggregate reconstruction, then integrate it with the already published H3 and FP32 refiners. Required final four-page PDF and Google Form take precedence over unexplained further training.
