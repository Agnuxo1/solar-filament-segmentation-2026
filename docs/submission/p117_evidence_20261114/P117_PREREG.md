# P117 FROZEN PREREGISTRATION - 2026-11-14 UTC
P116 verified 8 complete local shard hashes and 16 remote 4-KiB samples, not full remote checkpoint content. Hypothesis: all eight complete remote public shard SHA-256 digests match the public manifest, and concatenating the two public shards in each fold matches the four original checkpoint digests.
Immutable public ref: d1d88d6b16479af40dfcc4b734c193bcaa9a8203.
Frozen local P106 ref: 6843362b964dd62a53f19c975e777664c8ea2245.
Reference manifest SHA-256: 085e1011e6562404ba1943fe64373a23b769b38c01e806bf518e3063584e7e3d.
Use public unauthenticated HTTPS streaming GETs; require 200, correct complete length and SHA-256 for every part and assembled model. Limit to two parallel workers and eight public files (735580660 bytes total), memory chunks of 1 MiB, 45-second network timeout, eight-minute wall budget. No credential or API account use. Save errors and fail closed; do not retry access/rate limits. No large local file writes.
Success requires 8/8 public parts, 4/4 models, no errors; otherwise FAIL_OR_INCOMPLETE.
This is reproducibility of public model files only, not a TRAIN or TEST label metric, independent model inference, new PQ or candidate. H9b (submission 58112557, public PQ 0.39) must not be replaced or uploaded. No CUDA, training, or Kaggle actions. JEV/Goldfish not consulted (deterministic hashing). Independent comparator: frozen manifest and P116 verified local hashes.
