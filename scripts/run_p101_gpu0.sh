#!/usr/bin/env bash
# P101: serial, preregistered, GPU0-only CuDNN mode sensitivity experiment.
set -euo pipefail
ROOT=/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112
REPO="$ROOT/release_checkout_20261112"
OUT="$ROOT/p101_cudnn_h3_20261113"
INPUT=/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images
GPU=GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14
PYTHON=/home/ai/venvs/solarfil/bin/python
if [[ "${P101_LOCK_HELD:-0}" != 1 ]]; then
  export P101_LOCK_HELD=1
  exec /usr/bin/flock -x -w 60 "$ROOT/supervisor.lock" "$0" "$@"
fi
mkdir -p "$OUT"
exec >> "$OUT/p101_run.log" 2>&1
echo "P101_BEGIN $(date -u +%FT%TZ)"
[[ "$(date -u +%Y%m%d%H%M)" < 202611150000 ]] || exit 2
[[ "${CUDA_VISIBLE_DEVICES:-}" == "$GPU" ]] || exit 3
test -f "$REPO/docs/submission/P101_CUDNN_H3_PRECISION_PREREG_20261113.md"
echo "PREREG_SHA256 $(sha256sum "$REPO/docs/submission/P101_CUDNN_H3_PRECISION_PREREG_20261113.md")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PYTHON" "$REPO/scripts/p101_cudnn_h3_variants.py" --self-test
"$PYTHON" "$REPO/scripts/p101_cudnn_h3_audit.py" --self-test --out "$OUT" --images "$INPUT" --baseline "$ROOT/p100_h3_repeatability_20261113/a/F1" --historical "$ROOT/p63_h3_raw_replay/F1" --receipt "$OUT/P101_FINAL_AUDIT.json"
for variant in benchmark_off deterministic; do
  if [[ -f "$OUT/${variant}_receipt.json" ]]; then
    echo "P101_SKIP_COMPLETE $variant"; continue
  fi
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -F "$GPU"; then
    echo "P101_REFUSE_BUSY_GPU0"; exit 4
  fi
  echo "P101_START $variant $(date -u +%FT%TZ)"
  timeout --signal=INT 150 "$PYTHON" -u "$REPO/scripts/p101_cudnn_h3_variants.py" --variant "$variant" --images "$INPUT" --out "$OUT"
  echo "P101_DONE $variant $(date -u +%FT%TZ)"
done
"$PYTHON" -u "$REPO/scripts/p101_cudnn_h3_audit.py" --out "$OUT" --images "$INPUT" --baseline "$ROOT/p100_h3_repeatability_20261113/a/F1" --historical "$ROOT/p63_h3_raw_replay/F1" --receipt "$OUT/P101_FINAL_AUDIT.json"
sha256sum "$OUT/P101_FINAL_AUDIT.json"
echo "P101_COMPLETE $(date -u +%FT%TZ)"
