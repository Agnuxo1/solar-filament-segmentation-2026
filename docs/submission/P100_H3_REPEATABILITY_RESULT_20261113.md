# P100 — Exact repeated GPU0 H3 public-model inference across four folds

**Run date:** 2026-11-13 UTC.  
**Frozen protocol:** [P100 preregistration](P100_H3_REPEATABILITY_PREREG_20261113.md).  
**Runner:** [serial CUDA script](../../scripts/run_p100_gpu0.sh).  
**Independent audit:** [P100 auditor](../../scripts/p100_h3_repeatability_audit.py).  
**Raw machine receipt:** [P100_FINAL_AUDIT_20261113.json](P100_FINAL_AUDIT_20261113.json).

## Verified result

Eight serial GPU0 inference executions, two independent invocations (A and B)
for each of four frozen model folds, processed the same four lexically first
unlabelled official test images. All 32 generated H3 map files were checked
against their own immutable run receipts, source-image hashes, and previously
released public inference output P77 (F1) / P79 (F2-F4).

| Comparison | Exactly matching image/model pairs | Changed uint8 pixels |
|---|---:|---:|
| A versus B | **16/16** | **0** |
| A versus earlier public replay | **16/16** | **0** |
| B versus earlier public replay | **16/16** | **0** |

Both invocations reproduce every output byte of the existing public replay,
on this fixed pilot. This supports the interpretation that the known sparse
one-byte historical H3 discrepancies are not ordinary run-to-run variation
under the unchanged current GPU0/CUDA environment. This result does **not**
identify their cause or establish exact parity with the historical H3 cache.
No labels, PQ metric, leaderboard feedback or new Kaggle submission were used.

## Execution controls and recovery

The preregistration was committed publicly *before* any new GPU inference.
The run acquired the existing supervisor flock, verified nvidia-smi compute
PIDs, isolated GPU0 by UUID, and ran **one** CUDA inference at a time.
GPU1, which was occupied by an independent project, was not modified.
The systemd service had a 20-minute wallclock limit and 8 GiB RAM cap;
generated files and logs were stored on the NAS HDD.

One post-inference audit implementation error was caught: a NumPy
`bool_` was passed to JSON serialization. The typed Boolean conversion was
corrected, the incomplete temporary JSON was discarded, and **only the
CPU-only audit was re-executed**. The 32 verified CUDA map outputs were not
rerun or altered. The final audit passed its deliberate one-pixel corruption
and shape mismatch self-tests.

**Final receipt SHA-256:** `908d937517df8089522c8ee7a19fc1aadb0cab8ea6e8fe3b25c25491e44f4b8b`.

## Scope and next gate

This is a 4-image-per-fold pilot, not all 180 official test images. The
preregistered null result is complete for this subset. Follow-up study P101
should compare a bounded set of deterministic CuDNN and unautocast FP32
inference settings against the original historical H3 map and exact public
baseline without touching any labels or modifying pretrained weights.

The final delivery remains incomplete until a **clean raw-JPEG-to-CSV notebook**
and an **actual Google Form submission receipt** exist. The P100 result is
not a new official PQ score and must not be used to promote a Kaggle candidate.
