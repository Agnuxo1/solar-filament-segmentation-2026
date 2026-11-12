# P86 — Original FP32 public-source GPU-to-CPU detector parity pilot

**Preregistered:** 12 November 2026, before any P86 inference.  
**Question:** Does removing FP16 weight rounding restore exact public detector proposals when inference runs on the isolated RTX 3090 GPU0, relative to the frozen byte-exact P69 CPU reference?

## Inputs and isolation

Use only 180 original **unlabelled** competition test JPEGs as an immutable inventory; select the first five filenames in lexicographical order. Use each of the four released original FP32 Mask R-CNN checkpoints, reconstructed on the research HDD and checked against its signed SHA-256. Compare output against the immutable original CPU P69 reference per fold and image. No test masks, ground truth, leaderboard feedback, Kaggle upload, training or calibration are permitted.

All CUDA work must run on GPU0 UUID `GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14` under the pre-existing supervisor lock and only when `nvidia-smi` reports no active compute process for that UUID. GPU1 is reserved. Use no large writes on `/mnt/datos`.

## Frozen protocol, stop rule, budget

Run four folds × five images × two views, using unchanged geometric post-processing and full FP32 weights. Disable CUDA TF32 and CUDA autocast. Compare ordered proposal counts, score values, bounding box integer coordinates and entire crop-mask byte arrays. Preserve every fold's positive and negative outcomes. The primary outcome is the count of fully exact two-view images out of 20. Report separate same-index exact box, score, and mask counts, without IoU matching.

A full-domain GPU experiment may be separately preregistered **only if** all 20 pilots match exactly. Otherwise stop and investigate CPU/CUDA numerical differences; never repeat P82 or use the discrepancy to tune against Kaggle. External runtime cap: 600 seconds; one Python CUDA process, exclusive supervisor lock. Store a unique JSON receipt on the 985-GiB-free HDD with input and weight SHA-256, physical nvidia-smi preflight, code SHA-256, timestamps, and all negative findings. Stop immediately for hash failures, unexpected inventory, GPU conflict or malformed proposals.

**Priority context:** This is a bounded reproducibility experiment, not a candidate to replace the accepted H9b submission. Completion of the final four-page PDF, complete notebook and mandatory form remains the higher-priority deliverable.
