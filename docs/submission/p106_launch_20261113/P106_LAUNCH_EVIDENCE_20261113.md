# P106 — Public notebook execution launched

**Status: running at the snapshot of 13 November 2026, 14:08:09 UTC.**

P106 executes all eight code cells of the published raw-JPEG reconstruction notebook from a separate, clean worktree pinned to commit `6843362b964dd62a53f19c975e777664c8ea2245`. The execution started at **14:05:32.666833 UTC** under the persistent user service `solarfil-p106-public-clean-6843362.service`.

The input consists of the 180 authorised, original competition test JPEGs and the public model assets. A new output directory is used. Historical intermediate predictions are not inference inputs. The accepted submission and P103 output are configured only as final, post-inference comparison references. No test annotations, retraining or Kaggle submission is part of this execution.

## Verified preparation

- The checkout was clean and its HEAD matched the pinned commit before launch.
- All ten checkpoint files and all eight original-FP32 detector shards passed SHA-256 verification.
- All eight notebook code cells compiled, and fifteen required public source files were present.
- All thirteen package versions specified in the published requirements matched the final execution environment.
- Notebook SHA-256: `317b1781101fd6652b21b3f86d297b27136bd5c61ef7addfc8a904a7776314ad`.
- Execution-runner SHA-256: `50ec7a19344d76a74ef90109cbc40793a28a35ff00887e2929cc72b9f017a334`.

## Environment scope

The P106 virtual environment is separate, but it **reuses the verified scientific packages from the existing laboratory environment through a read-only package path**. It is therefore a clean code checkout and an isolated run, not a demonstration of installing the entire dependency stack from scratch.

The original environment lacked pytest 9.1.1 and exposed segmentation-models-pytorch 0.5.0. P106 locally installs pytest 9.1.1, its missing dependencies, and the required segmentation-models-pytorch 0.6.0 without modifying the shared environment. Exact versions are recorded in `environment_versions.json`.

## Live execution evidence

At **14:08:09 UTC**, the service was active/running with parent PID **2363459**. The designated RTX 3090 GPU0 showed **100% utilisation and 3,114 MiB VRAM**, with P106 CUDA process PID **2365085**. The output inventory contained **180/180 H3 maps for F1 and 86/180 for F2**; F3 and F4 had not yet started. Two notebook code cells were complete and the H3 cell was executing.

The published configuration is retained: CUDA for native H3 and original-FP32 detector inference; CPU FP32 for both crop refiners. GPU1 and the existing supervisor are left unchanged.

The service records progress and errors, updates an executed-notebook copy, performs the final COCO-RLE geometry checks, and compares the completed CSV against the accepted upload and P103. It has a maximum duration of eight hours, a 200% CPU quota and a 10 GiB RAM limit.

**This snapshot establishes a real launch and measured progress, not completion, exact historical-CSV parity, or a new Panoptic Quality score.** The official PQ 0.39 belongs to submission 58112557 and is not attributed to P106.

The four JSON copies are byte-preserving snapshots of the server artifacts. `capture_metadata.json` records their hashes and the timestamped GPU/service observation.

