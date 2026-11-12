# P77 - full public H3 source-to-map reproducibility study

Preregistered 2026-11-12 UTC before starting the complete 180-image run. The existing P76 single-image smoke test established only stage-one loading. P77 tests its full-domain repeatability using *exactly the same frozen public model source and F1 model weights*, without changing inference parameters or accessing labels.

## Hypothesis and fixed design
Every one of the 180 original unlabelled Kaggle TEST H-alpha JPEGs can be transformed to a valid 2048 x 2048 uint8 probability PNG by the public native_model.NativeUNet and frozen F1 ResNet34 fp16 weights. Input filenames, model weights, script hashes, output map hashes and elapsed time will be recorded. This is a reproducibility/engineering experiment, not a PQ optimization, not a new model, and not the accepted final H9b CSV reconstruction.

Primary PASS: all 180 files accounted for, exactly one PNG per input, strict weight hash verification, CUDA access restricted to GPU0 UUID GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14, and a durable receipt confirming test_labels_read=false. Failure: nonzero exit, incorrect image count/shape/hash, overlap with CUDA process, or missing receipt. Resume interrupted output through the public script --resume; never overwrite a completed successful receipt.

The experiment uses the same supervisor.lock exclusive lock as the original systemd experiment supervisor; it may delay an F2-F4 fold-evaluation stage but cannot start a second GPU0 process while holding this lock. CPU-only systemd scoring can coexist, memory capped 8G and CPU capped 200%; GPU1 is never allocated. If another client occupies GPU0 when the lock is acquired, fail closed and do not retry automatically. Run has a 2700-second bound; output and logs are restricted to the large HDD, not /mnt/datos.

Afterward, compare output SHA256 to historical H3 maps in a separate, explicitly registered audit without changing parameters. Report full-domain byte-level parity honestly and do not conflate near identity of maps with full final CSV fidelity or Kaggle PQ. No Kaggle submission in this protocol. The pending four-page report/form/notebook remain higher priority for final competition eligibility.

JEV/Goldfish: no new scientific training hypothesis is promoted by P77; previous OpenClaw lead recommended focusing on public reproducibility. No new JEV or Goldfish execution is claimed.
