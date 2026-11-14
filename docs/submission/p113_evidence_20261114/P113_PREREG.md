# P113 ? Preregistered independent RLE integrity audit (2026-11-14 UTC)

Hypothesis: the genuine accepted H9b CSV and the clean P106 rebuild use structurally valid compressed COCO RLE for every output instance, and independent count-stream decoding agrees with pycocotools on every foreground area. Differences between predictions must not be attributed to malformed serialization without such evidence.

Scope: read-only H9b and P106 CSVs, 180 original TEST JPEG headers (dimensions only), no test annotations, no selector fitting, no neural inference, no model training, no Kaggle upload. Group by physical image stem; examine all rows, not a sample.

Locked checks: exact CSV headers; unique filament IDs; image IDs in original JPEG inventory; all 180 images present; exactly equal per-image IDs/counts across both CSVs; validate ASCII compressed counts, finite/nonnegative integer counts and total pixels H*W; foreground area from independently decoded alternating counts equals pycocotools.mask.area; hash both inputs and report area-difference statistics by identical filament IDs. Any failure aborts and reports errors without modifying submitted CSVs.

Budget: CPU only, maximum 180 seconds. Deterministic; random seed not applicable. Output: full JSON receipt and source script on project HDD. No PQ/IoU/accuracy claims without ground truth.

Promotion gate: this is a serialization-integrity test ONLY. No score or candidate promotion, no final-selection change, no modification of organizer submission, no link or credential publication.
