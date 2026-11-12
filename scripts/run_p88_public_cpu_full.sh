#!/usr/bin/env bash
# P88: low-priority, CPU-only full-domain public-source parity replay.
set -euo pipefail
ROOT=/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112
OUT="$ROOT/p88_public_fp32_cpu_full_20261112"
mkdir -p "$OUT"
exec >> "$OUT/runner.log" 2>&1
echo "P88_BEGIN $(date -u +%FT%TZ)"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$ROOT/release_checkout_20261112/release/source:$ROOT/release_checkout_20261112/scripts"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt 20971520
flock -n "$ROOT/p88_cpu.lock" timeout --signal=INT 43200 \
  /home/ai/venvs/solarfil/bin/python -u \
  "$ROOT/release_checkout_20261112/scripts/p88_public_fp32_cpu_full.py" \
  --images /home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --weights "$ROOT/p85_fp32_smoke" \
  --reference "$ROOT/p69_cpu_exact_reference_replay" \
  --out "$OUT"
echo "P88_EXIT_OK $(date -u +%FT%TZ)"
