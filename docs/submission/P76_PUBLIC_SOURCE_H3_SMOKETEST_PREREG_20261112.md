# P76: public-source H3 reconstruction smoke test (preregistered 12 Nov 2026)

**Purpose.** The existing local P63 study reprocessed 720 test images through the server's private H3 model module. The public release did not include that exact one-channel model definition. This independent test begins from **the exact model source now added to the public release**, its previously published weight and the original unlabelled competition image, and checks that an external researcher can regenerate a native-resolution map. This is a packaging/reproducibility test, not a new PQ experiment.

**Frozen design, before execution.** First JPEG in lexical order of the official 180-image test set; F1 ResNet34-U-Net; published F1 fp16 state with verified public manifest SHA256; original 2048x2048 grayscale input; solar disk normalization radius 950, eight fixed D4 transformations, CUDA fp16 autocast and uint8 rounded 0..255 output. No test labels, calibration masks, hidden leaderboard scores or external images. The entire 180-image count is verified, but only the predetermined first image is inferred in this bounded smoke test.

**Primary gate.** Model loads strictly using only published source/weights, returns one 2048x2048 uint8 PNG, a per-image SHA256 and reproducible receipt. Secondary descriptive parity checks will compare this new map bytewise with the existing P63 output and the frozen historical H3 map for the same image, without changing any settings. Success of the smoke test **does not** establish end-to-end H9b CSV parity, scientific accuracy gains or bytewise identity for all 180 images.

**Resource and execution contract.** One process on NVIDIA GPU0 UUID GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14 only when nvidia-smi confirms no competing CUDA process; GPU1 must remain free. Max GPU allocation 75% of VRAM, output and logs only on NAS HDD, five-minute execution budget, no replacement of published weights. Existing systemd supervisor remains authoritative for its own queue; if its CUDA job has started, this smoke test is deferred. Do not retry repeatedly to occupy the GPU.

**Judgment response.** The OpenClaw research-lead review of an earlier FP32-vs-fp16 idea returned MODIFY and prioritized the public full-pipeline notebook and final Google Form. This narrower source-packaging test implements that priority. The review explicitly did not execute JEV or discover a Google Goldfish skill; neither action is claimed as completed here.

**Delivery gate.** The result may justify an additional public stage-one notebook section, but the public release must still separately cover Mask R-CNN proposals, H6 selection, crop refiners and final CSV assembly. No Kaggle submission until the stronger reproducibility and independent validation gates are satisfied.

