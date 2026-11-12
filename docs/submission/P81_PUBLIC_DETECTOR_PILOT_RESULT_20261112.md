# P81: public FP16 detector pilot and source repair (12 November 2026 UTC)

## Preregistration and scope

The [frozen P81 protocol](P81_PUBLIC_DETECTOR_PILOT_PREREG_20261112.md) selected one original unlabelled test JPEG before running, with two views per public FP16 Mask R-CNN checkpoint across F1–F4. Input: 20110120105534Ch.jpeg, SHA-256 7190e95661c142b194c3d2afddfedfee3382c2b3ded2f2f1b023d13e3bd58baf.

The first attempted run failed during import, without activating CUDA: the published filseg.common module lacked autocast, make_scaler, load_state, save_state, and amp_dtype. We restored exactly those historical original functions from the author's archived source, reviewed the 37-line additive diff, compiled and imported the public detector, then reran the unchanged protocol and weights. No input, threshold or scoring rule was adapted.

## Actual GPU pilot results

The repaired public implementation strictly loaded all four SHA-256-verified FP16 checkpoints, generated original and horizontally reflected mask proposals, validated crop geometry, and saved four atomic output records on the NAS HDD. GPU0 UUID GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14 was assigned, GPU1 was left untouched. A shared file lock serialized the pilot with the existing supervisor. Peak allocated CUDA memory: 1,229,842,944 bytes (1.145 GiB); detector execution: 5.91 seconds, excluding service startup.

| Fold | Public proposals (original / reflected) | Historical FP32 counts | Same-index exact boxes | Exact same-index crops |
| --- | ---: | ---: | ---: | ---: |
| F1 | 10 / 14 | 10 / 14 | 24 / 24 | 0 |
| F2 | 19 / 16 | 19 / 16 | 32 / 35 | 0 |
| F3 | 14 / 21 | 14 / 21 | 33 / 35 | 0 |
| F4 | 15 / 13 | 15 / 13 | 26 / 28 | 0 |
| **Total** | **122** | **122** | **115 / 122** | **0 / 122** |

The public FP16 checkpoint values loaded into FP32 model parameters preserved all 122 proposal counts on this one image but did not reproduce any historic FP32 proposal crop byte-for-byte; seven boxes differ. **Exact final H9b submission parity is not established.** This one-image smoke check is not evidence of equivalence over 180 images, improved PQ or a Kaggle score. No test labels, calibration optimization or new Kaggle submissions were involved.

Private receipt: paired-small-20261112/p81_public_detector_pilot/receipt.json, including source/weight/input/output hashes. The source code is [replay_public_detector_p81.py](../../scripts/replay_public_detector_p81.py). Reproduce the offline reflection test from a public repository checkout:

    PYTHONPATH=release/source python scripts/replay_public_detector_p81.py --self-test --release release --images /path/to/test_images --out /large-hdd/p81

The optional --reference argument should be used only with trusted, locally generated pickle files; untrusted pickles can execute code when deserialized. For GPU inference use exclusive access to GPU0 and the supervisor lock, as demonstrated by the private systemd wrapper.

The historically accepted Kaggle submission **58112557** remains at public PQ **0.39**. The [four-page English report](TECHNICAL_REPORT_4P_20261112.pdf) exists. A completely replayed original-image-to-final-CSV notebook, exact accepted CSV comparison and actual [mandatory final Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform) confirmation remain open.
