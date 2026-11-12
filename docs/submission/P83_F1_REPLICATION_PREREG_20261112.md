# P83: F1 independent-seed replication ? frozen protocol

Frozen on 12 November 2026, before evaluating the three already trained F1 seed checkpoints.

**Question:** Does the small-centred sampling arm consistently outperform the paired control when seed variance is measured? Earlier F1?F3 test results show inconsistent effects and confidence intervals including zero, so this is a secondary replication and is **not** a fresh blind test.

**Design:** seeds 20261113, 20261114, and 20261115, each saved at 3600 and 7200 steps, both treatment and control. Read only the pre-existing official TRAIN-derived F1 caches for inference; F1 117-image calibration and 139-image holdout use the immutable F1 split. All seed outputs use the **original P59 F1 calibration policy**, fixed before any new seed comparisons. Record every native PQ result, counts, paired bootstrap CI, run status and hashes, without selection on test.

**Resources:** one GPU UUID GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14 only for GPU inference; CPU PQ workers have a single 12 GB, 180% CPU slot. Outputs remain on the NAS 2 TB HDD; GPU1 and the full /mnt/datos SSD must not be used for large writes. The existing hourly-independent systemd timer serializes stages and resumes from receipts. Freeze at 2026-11-15 00:00 UTC.

**Interpretation:** a positive mean alone does not promote a candidate. Because F1 test outcomes have already been viewed, this study cannot be claimed as independent confirmation. It will never trigger a Kaggle submission automatically. The goal is robust negative/positive replication and accurate reporting.

**JEV:** Remote plan ID 1794510498-design suggested main agent, high scientific scrutiny and a second opinion. A direct second-opinion query returned a local-schema error; no fabricated expert endorsement. Goldfish review was not requested by router.
