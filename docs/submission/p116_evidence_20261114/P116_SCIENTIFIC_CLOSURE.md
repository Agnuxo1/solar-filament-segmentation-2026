# P116 scientific closure — independent public-release byte-integrity controls

**Executed:** 14 November 2026, after SHA-256-frozen preregistration.
**Outcome:** `PASS_PUBLIC_SAMPLED_BYTES_AND_LOCAL_FULL_HASHES`.
**Wall time:** 20.935 seconds. **Runtime:** CPU and public unauthenticated HTTPS only.

## Motivation and scope

The previous P115 audit established HTTP HEAD availability for twelve public release assets; HEAD response codes alone do not prove that the published bytes correspond to the frozen local scientific release. P116 uses content-bearing GET requests and local complete-file hashing to strengthen this evidence. The protocol was fixed in `P116_PREREG.md` (SHA-256 `d952119bb1d0000e19edfc880a298cfe022a516c709ae71f4b38a8c59d85334c`) before any P116 data requests. Source SHA-256: `f3dccd23490e612b8c5af4b212bc3cfa57e995082bcb9529e1743865728e8780`.

The test anchored all public GET requests to the immutable GitHub release commit `d1d88d6b16479af40dfcc4b734c193bcaa9a8203` and independently read local reference files from the clean P106 checkout, commit `6843362b964dd62a53f19c975e777664c8ea2245`.

## Results and reproducible evidence

| Preregistered gate | Observed result |
| --- | --- |
| Eight local original FP32 Mask R-CNN shards: exact file sizes and full SHA-256 against public manifest | **8/8 PASS** |
| Four locally assembled model streams: full SHA-256 against documented original checkpoints | **4/4 PASS** |
| Unauthenticated public HTTP 206 byte-range GETs: first and last 4,096 bytes of every shard, with exact total-size Content-Range and local byte comparison | **16/16 PASS** |
| Complete public HTTP GET plus full SHA-256 equality: requirements.txt, inference notebook, and shard manifest | **3/3 PASS** |
| Complete public four-page R2 report download plus recorded SHA-256 `8f75a47996fa7e6e5f37f5d8e85a6cf5b64033701507782a54a81d786e5ed9f5` | **1/1 PASS** |
| Errors and HTTP semantic failures | **0** |

Full machine-readable observation records and their individual hashes are in `P116_FINAL_RECEIPT.json` (SHA-256 `2f7d85bde20e23046e8f994e771abf2a34eb017d05e93bc81180d02e6c6b8257`). The complete runtime log is `P116_RUN.log` (SHA-256 `816e4d07c2520f847d23069be42c59a71358e1546eeb0d2a6a1074490500ae0c`). The standalone `p116_public_byte_integrity.py` requires only the Python standard library and reproduces the method with the frozen local release files.

## Interpretation and limitations

P116 **does verify full local checkpoint integrity** and the **exact public first and last 4 KiB** of each of eight shards. It **does not download or hash the entire public version of every large shard**. Matching two small byte ranges cannot prove complete public-shard equivalence. The four small public assets were downloaded completely and compared using SHA-256. The audit does not independently reproduce inference on separate hardware and cannot establish model accuracy or Panoptic Quality.

No TRAIN or TEST annotations, Kaggle API, protected browser context, private files, training, GPU compute, hidden-label tuning or leaderboard probing were used. JEV and Goldfish were not consulted because the task was a deterministic release-validation protocol.

The accepted H9b artifact (submission **58112557**, public **PQ 0.39**, genuine CSV SHA-256 `24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60`) remains the selected final submission. This audit does not transfer that score to P106, generate a new candidate, change the organizer response, or justify another Kaggle upload. The P108 outside-fold provenance gate still blocks the proposed 1.15× context ablation.

**Release decision:** Preserve the original form, report R2, notebook, weights and selected H9b artifact. P116 is supplementary evidence of public access and reproducible file contents only.
