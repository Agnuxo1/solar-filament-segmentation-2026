# P117 scientific closure - full public checkpoint SHA-256 verification

**Executed:** 14 November 2026, 11:03:24 to 11:03:43 UTC.
**Outcome:** PASS_FULL_PUBLIC_8_PARTS_4_MODELS, 18.866 seconds.
**Scope:** unauthenticated public HTTPS, CPU-only streaming SHA-256; reproducibility of published model bytes.

## Why the experiment was needed
P116 proved SHA-256 of all local original Mask R-CNN checkpoint shards, checked the four local reassembled checkpoint streams, and compared only the first and last 4 KiB of the large publicly served shards. That sampled test could not establish whole-file identity. P117 preregistered full public file hashing before network reads, using the immutable GitHub commit d1d88d6b16479af40dfcc4b734c193bcaa9a8203.

## Preregistered tests and observed results

| Frozen gate | Observation | Result |
| --- | --- | --- |
| Full public SHA-256 of all eight Mask R-CNN FP32 shards | 8/8 complete streams match the frozen manifest | PASS |
| Concatenated public checkpoint hashes for four folds | 4/4 original model SHA-256 digests and sizes match | PASS |
| Total bytes verified over HTTPS | 735,580,660 bytes | PASS |
| Execution controls | Two concurrent workers, 1 MiB memory chunks, no large file cached | PASS |
| Error log | Empty | PASS |

Each of the four models contains 183,895,165 bytes. All twelve expected content digests (eight shards and four full checkpoint streams) match the immutable manifest, whose SHA-256 is 085e1011e6562404ba1943fe64373a23b769b38c01e806bf518e3063584e7e3d. This closes the **public full-byte-integrity gap** explicitly left open by P116. It does not demonstrate a second independent inference run on different hardware.

## Reproducible evidence

- Frozen preregistration: P117_PREREG.md, SHA-256 db938dd17bb101704f8845cfcf6cd775448030d7811524b0050ca85f4d5f74ca.
- Complete Python program: p117_public_full_hash.py, SHA-256 b0f9a96795179a7aef9d25740d5299f65d66fd2f602769a6e8b1211a83bca5de.
- Complete machine-readable observation receipt: P117_FINAL_RECEIPT.json, SHA-256 cac6f6c26255bd80659d0629981b10b16ac4b67eb2ad79cef3e25ddba302827d.
- Execution log: P117_RUN.log, SHA-256 3d4b85e5e7157ca7ac44c0169482222d38286d738ba73389ad0126e8194558ef.
- The clean P106 local reference source remained pinned to 6843362b964dd62a53f19c975e777664c8ea2245, and was not changed.

## Scientific interpretation and safeguards

P117 establishes that the entire four-model FP32 detector release is downloadable in exactly the bytes committed by the public manifest. It does **not** use ground-truth labels, measure performance on TRAIN or TEST, infer new masks, prove model accuracy, establish Panoptic Quality, or upgrade the official score of P106. It does not remove the P108 out-of-fold provenance block on the proposed 1.15x TRAIN ablation.

The genuine H9b accepted Kaggle submission **58112557**, originally confirmed public PQ **0.39**, remains selected; genuine CSV SHA-256 **24b3ea1c8e151c3053c69e61bc3d0a0a09bd7171fe82c95f7c109ce407efdc60**. No new Kaggle submission, Google Form response, PDF change, notebook change, GPU training, CUDA job, protected browser action or account switch was performed. JEV/Goldfish were not consulted for this deterministic hash exercise. The only writes during execution were the small preregistration, code, receipt, and log on the project HDD.

**Release decision:** Keep the official H9b submission, the four-page R2 report, notebook and previously acknowledged organizer response unchanged; P117 is supplementary public-release evidence.
