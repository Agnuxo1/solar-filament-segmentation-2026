# P119  Independent final-release CPU gate (preregistered)
Scope: independent source checkout at public GitHub commit 973f17da9ddd1a227a46047443d74995102956fa, original public release files.
Hypothesis: latest public code retains runtime, resume and weights-integrity contracts confirmed before its publication.
Inputs: public source tree and released weights; no hidden test labels; accepted H9b remains read-only.
Method: Python 3.13 unit-test discovery tests/test_public_replay_runtime.py; scripts/preflight_h9b_clean_notebook.py --root . --full-hash; source-digest and official artifact checks.
Budget: 10 CPU minutes, zero GPU; preregistered before launching tests.
Accept if all unit tests pass, notebook compiles and every declared asset hash matches. No precision/PQ claim or Kaggle upload can follow solely from this gate.
Provenance: reused P106 Python scientific dependency installation (verified versions), not an independent clean package install; new worktree, commands and result logs are separate.
Codex/Claude agents: not considered available without direct executor evidence; this is an assistant-run independent release audit, not an agent-to-agent handoff completion.
Storage: /srv/almacen/nas-2tb-1 (889 GiB free at 2026-11-14T16:02:49Z), selected over /srv/almacen/nas-2tb-2 (648 GiB), /mnt/datos (8.4 GiB) and / (17 GiB). Environment caches and temp confined to this volume.
