# P80: preregistered F2-F4 full-domain public-source map parity

Registered on 2026-11-12 UTC **before comparing any F2-F4 public-vs-P63 full-domain maps**. The P79 original-image replay uses the same unmodified public `native_model.py` and published F2/F3/F4 checkpoints on the 180 original unlabelled Kaggle test JPEGs per fold. This P80 *CPU-only*, read-only comparison evaluates quantized prediction reproducibility, not predictive accuracy, PQ, or Kaggle submission score.

The three comparisons are fixed as F2, F3, F4, always all 180 image names and 2048x2048 uint8 maps each. Reference is the untouched P63 private stage-one probability map cache of matching fold. The test algorithm is exactly the already published `scripts/audit_h3_full_domain_parity.py`, with each P79 per-image SHA256-verified receipt, one immutable P80 JSON output per fold, no threshold/parameter modification and no hidden test labels.

The fixed per-fold descriptive parity gate is >=99.99% exactly matching quantized pixels and maximum absolute byte difference <=1. Report numerical outcomes and gate failures honestly, with no retries or source modifications to meet the gate. A failing gate is a source parity discrepancy for investigation, not a change in PQ or leaderboard rank.

Execution runs sequentially on CPU without CUDA, separate from systemd GPU0 supervisor. 2 GiB RAM cap, 75% CPU cap, one minute nominal budget per fold, all audit outputs to NAS HDD. Run the embedded arithmetic self-test before processing F2. Never overwrite verified outputs.

Publish the three machine receipts, an English summary and protocol with appropriate caveats. This study cannot establish full H9b RLE CSV reproducibility or eligibility of the final Google Form. No new JEV/Goldfish consultation is claimed.
