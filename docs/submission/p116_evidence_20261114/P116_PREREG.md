# P116 preregistration — independent public-release byte-range verification

Status: PREREGISTERED; no P116 experiment executed at time of writing.
Date: 2026-11-14 UTC.

## Motivating observation
P115 checked unauthenticated HTTP HEAD availability (12/12 status 200), but did not retrieve or compare served bytes. A successful HEAD may not establish correct public artifact content. P106 separately verified local model weights and executed the notebook.

## Frozen hypotheses and acceptance criteria
H1: The four published original FP32 Mask R-CNN checkpoints, split into eight shards, still match their documented SHA-256 and exact sizes **locally**.
H2: An unauthenticated public HTTP GET with Range support returns status 206, the correct Content-Range total size, and exactly matching first and last 4096 bytes of each of the eight shards, anchored to GitHub commit d1d88d6b16479af40dfcc4b734c193bcaa9a8203.
H3: The public notebook, requirements and shard manifest are fully downloadable and byte-identical to frozen local release files; report R2 is fully downloadable with SHA-256 8f75a47996fa7e6e5f37f5d8e85a6cf5b64033701507782a54a81d786e5ed9f5.

## Frozen inputs and controls
- Public repository: Agnuxo1/solar-filament-segmentation-2026.
- Immutable GitHub commit: d1d88d6b16479af40dfcc4b734c193bcaa9a8203.
- Frozen verified checkout: p106_checkout_6843362 with source commit 6843362b964dd62a53f19c975e777664c8ea2245.
- Local manifest: release/models/fp32_shards/manifest.json, all eight parts, with exact sizes and full SHA-256.
- Original report R2 SHA-256 from the documented receipt.
- Tests operate on published model bytes only, with no test labels, model training or CUDA.
- Reference genuine H9b hash: 24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60, unchanged and not uploaded.
- Do not assume that matching sampled ranges proves complete public weight equivalence.

## Procedure
1. Verify all eight local shard sizes and full SHA-256 against the public manifest, plus each assembled checkpoint SHA-256 using streaming hashing.
2. For each shard, request first and last 4096 bytes over public HTTPS, reject anything but HTTP 206 and a correct Content-Range and byte count, and verify exact local byte equality.
3. Retrieve requirements.txt, the notebook, the shard manifest and the four-page R2 PDF fully with public GET; compare full SHA-256 to local frozen files or the documented R2 digest.
4. Produce a machine-readable receipt with each result, failure, UTC timestamps and immutable GitHub commit.
5. Reject promotion of any model / Kaggle candidate. This is release-integrity evidence only, not an accuracy or PQ experiment. Leave the final H9b selection, reports, form and weights unmodified.

## Boundaries
Budget: 180 seconds wall-clock, CPU only, 4 workers maximum for short HTTPS requests (timeouts 20 seconds); no oversized response (>5 MiB) on the four small file downloads; no complete public shard download; no code execution from the downloaded content. Abort or mark FAIL on mismatches, network failure or HTTP semantics violation. Never substitute a 200 response for a requested byte range. No JEV/Goldfish consultation is necessary for deterministic file validation; record that accurately. Independent review: checks use two independently sourced digests, published manifest and receipt.
