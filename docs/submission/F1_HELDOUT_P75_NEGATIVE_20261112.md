# Solar Filament 2026: held-out F1 and P75 audit (12 November 2026)

**Scope and evidence boundary.** These are independent local evaluations on a disjoint portion of the Kaggle-supplied **TRAIN** set, or inference efficiency measurements on unlabeled Kaggle TEST pixels. They are **not** newly awarded Kaggle public/private scores, and do not authorize a new submission.

## F1 paired small-filament sampling: held-out results

The F1 7,200-step P59 control/treatment experiment uses one frozen post-processing policy, selected on 117 calibration images (threshold 0.25, minimum area 200, peak 0.95, mean 0.7), followed by **139 held-out physical test images** and 2,000 resamples of those physical-image units. Test labels were not used to choose the policy, seeds or stopping points.

| Frozen model and arm | Test PQ | Delta versus 3,600-step control | Paired 95% CI |
|---|---:|---:|---|
| 3,600 control | 0.3975942962 | reference | NA |
| 3,600 treatment | 0.3949464780 | -0.0026478182 | [-0.0089731200, +0.0035992732] |
| 7,200 control | 0.3987962156 | +0.0012019194 | [-0.0029353326, +0.0056371552] |
| 7,200 treatment | 0.3940773218 | -0.0035169744 | [-0.0103631406, +0.0032028045] |

**Decision: HOLD/NO PROMOTION.** All three intervals include zero. Longer treatment training did not establish an improvement even versus its matched, weaker-family reference. The separate H9b calibration PQ of 0.462433 belongs to a different, historically used calibration cohort and cannot be presented as a direct comparison with this untouched F1 test. The original frozen H9b Kaggle submission remains the only verified accepted reference.

The test report is retained on IA-Server's research HDD as `paired-small-20261112/evaluations/test_pq_report.json`; full per-image details are not reproduced here. The scientific report expressly labels calibration, held-out test, and Kaggle scores separately.

## P75 preregistered GPU batching: terminated partial experiment

The preregistered single-RTX-3090 protocol compared unchanged eight-view F1 U-Net predictions over the first 16 lexically sorted, unlabeled official test images. Its reference batch 1 achieved 1.98763 images/s; batch 2 achieved 2.18958 images/s (about **+10.16%**) and batch 4 achieved 2.06877 images/s (about **+4.08%**). Maximum absolute per-pixel difference from the batch-1 quantized predictions was one byte for both altered batch sizes.

The protocol required **at least +15% speed** in both F1 and F2 *and* maximum quantized error no greater than one byte. Neither candidate cleared the F1 speed threshold. Before F2, the source aborted because its literal expected F2 SHA-256 had 63 instead of 64 characters; the immutable F2 published weight was independently verified against the public release manifest. The spelling error was corrected in the research script without changing any weights or experimental design, and the original source copy was preserved. **F2 was not measured.** There is no complete P75 result, no batch-size promotion and no reason to repeat the same hypothesis purely to occupy a GPU.

The append-only provenance is on the HDD in `p75_batched_gpu/P75_PARTIAL_RESULT_20261112.json`, with the original source SHA-256 `32c7cdc68b1f827985774aa2469309c4fb1e9ac067fbb3dedc2ba9192424de15`, corrected source SHA-256 `0b3ee2debcb7e1e6acb93af0f5508cabf2a0fbe552b91bd9ceee62e020756388`, and original preregistration SHA-256 `b61989a6a2004fbcd6bf50d1ea5dda6964af30119d37459ec287adbc25f0fc9a`.

## Submission and remaining work

The latest authenticated Kaggle CLI check on 12 November 2026 confirms **submission 58112557**, complete, **public PQ 0.39**, dated 8 November 2026. No submission was made in this study. Fold F2NAF4 evaluation is still proceeding sequentially under the existing systemd supervisor; their PQ values must not be guessed or claimed early. A reproducible, independently run original-image-to-final-CSV notebook and the [organizers' Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform) response confirmation remain essential outstanding delivery gates. The published four-page PDF and released weights are not equivalent to a verified final Google Form response.

No claim is made about power savings, CO2, water use, or full pipeline parity without direct measurements.

