# P118 — Pre-freeze public-release parity audit (preregistration)

Date: 2026-11-14 UTC. Audit scope: deterministic artifact parity, not model accuracy.

## Frozen hypothesis
The publicly released notebook and report R2 at Git commit 97f1b08703c90196fcd6f89e9e9988ba7aeb979f are byte-identical to the source notebook executed in P106 and the acknowledged organizer R2 file, while the genuine selected H9b CSV, P103 and P106 are locally immutable.

## Frozen references
- P106 notebook SHA256: 317b1781101fd6652b21b3f86d297b27136bd5c61ef7addfc8a904a7776314ad.
- Public report R2 PDF SHA256: 8f75a47996fa7e6e5f37f5d8e85a6cf5b64033701507782a54a81d786e5ed9f5.
- Public report R2 source ZIP SHA256: 6bee5b287bf6920ae8ef3bab82b7518a6366140da92de19b98e6b972b128955b.
- H9b selected CSV SHA256: 24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60.
- P103 CSV SHA256: 8ae3160af9bddd5ac3ef914bc807da71e3e9cbad0444cc055d2ff0056cdfe2ba.
- P106 CSV SHA256: 16f7dd60e8308d1cdccff1b6b16e80eff74bf08b5996d5a5d58c9ce295ebb9c3.
- P117 published scientific receipt SHA256: cac6f6c26255bd80659d0629981b10b16ac4b67eb2ad79cef3e25ddba302827d.

## Procedure and stopping rules
CPU only, standard-library Python, stream public HTTPS bytes into a disposable directory on the project HDD; compare SHA256 of all seven frozen artifacts. Validate eight notebook code-cell source digests against P106_FINAL_RECEIPT.json; inspect consistency of published organizer receipt and R2 manifest with frozen values. Check ZIP central-directory CRC integrity. Preserve a timestamped JSON receipt and program source; maximum wall time five minutes. No TRAIN or TEST labels, no GPU, no model training or inference, no Kaggle uploads, no form edits, no secrets. Zero random sampling; seed not applicable. Fail closed if a digest or any provenance gate differs. A pass is a reproducibility/integrity result, never a new PQ or a basis for promoting P106 over H9b. No external review claimed.