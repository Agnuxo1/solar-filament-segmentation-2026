#!/usr/bin/env bash
# P100: preregistered, serial, bounded CUDA inference on assigned GPU0 only.
set -euo pipefail
ROOT=/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112
REPO="$ROOT/release_checkout_20261112"
OUT="$ROOT/p100_h3_repeatability_20261113"
INPUT=/home/ai/pc/d/PROJECTS/196_NeuroPixel/kaggle/filament/data/MAGFiLO_1.0_Kaggle_2026/test/test_images
GPU=GPU-b2046dc8-91f7-fd8a-b237-96bb9e293d14
PYTHON=/home/ai/venvs/solarfil/bin/python

if [[ "${P100_LOCK_HELD:-0}" != 1 ]]; then
    export P100_LOCK_HELD=1
    exec /usr/bin/flock -x -w 60 "$ROOT/supervisor.lock" "$0" "$@"
fi
mkdir -p "$OUT"
exec >> "$OUT/p100_run.log" 2>&1
echo "P100_BEGIN $(date -u +%FT%TZ)"
[[ "$(date -u +%Y%m%d%H%M)" < 202611150000 ]] || { echo "P100_AFTER_FREEZE"; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "$GPU" ]] || { echo "P100_WRONG_GPU"; exit 2; }
[[ "$(find "$INPUT" -maxdepth 1 -name '*.jpeg' -type f | wc -l)" == 180 ]] || { echo "P100_WRONG_INPUT_COUNT"; exit 2; }
[[ -f "$REPO/docs/submission/P100_H3_REPEATABILITY_PREREG_20261113.md" ]] || exit 2
[[ -f "$REPO/scripts/p100_h3_repeatability_audit.py" ]] || exit 2
echo "PROTOCOL_SHA256 $(sha256sum "$REPO/docs/submission/P100_H3_REPEATABILITY_PREREG_20261113.md")"
echo "GPU_STATUS"; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
echo "CUDA_PROCESSES"; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -F "$GPU"; then
    echo "P100_REFUSING_OCCUPIED_GPU0"; exit 3
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PYTHON" "$REPO/scripts/p100_h3_repeatability_audit.py" --self-test --images "$INPUT" --out "$OUT" --ref-f1 "$ROOT/p77_full_h3_public_20261112" --ref-other "$ROOT/p79_full_h3_public_20261112" --receipt "$OUT/P100_FINAL_AUDIT.json"
for condition in a b; do
    for fold in F1 F2 F3 F4; do
        receipt="$OUT/$condition/${fold}_h3_replay_receipt.json"
        if [[ -f "$receipt" ]]; then
            echo "P100_SKIP_COMPLETED $condition $fold"; continue
        fi
        echo "P100_START $condition $fold $(date -u +%FT%TZ)"
        if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -F "$GPU"; then
            echo "P100_GPU0_BECAME_BUSY"; exit 3
        fi
        timeout --signal=INT 130 "$PYTHON" -u "$REPO/scripts/replay_h3_from_original_images.py" --images "$INPUT" --out "$OUT/$condition" --fold "$fold" --limit 4 --device cuda --gpu-uuid "$GPU" --resume
        echo "P100_DONE $condition $fold $(date -u +%FT%TZ)"
    done
done
"$PYTHON" -u "$REPO/scripts/p100_h3_repeatability_audit.py" --images "$INPUT" --out "$OUT" --ref-f1 "$ROOT/p77_full_h3_public_20261112" --ref-other "$ROOT/p79_full_h3_public_20261112" --receipt "$OUT/P100_FINAL_AUDIT.json"
sha256sum "$OUT/P100_FINAL_AUDIT.json"
echo "P100_COMPLETE $(date -u +%FT%TZ)"
