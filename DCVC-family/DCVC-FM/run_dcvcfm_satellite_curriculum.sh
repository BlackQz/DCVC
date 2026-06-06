#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=""
MODEL_I="checkpoints/cvpr2024_image.pth.tar"
MODEL_P="checkpoints/cvpr2024_video.pth.tar"
OUT_ROOT="checkpoints/dcvcfm_satellite_curriculum"
DEVICE="cuda"
IMG_H=256
IMG_W=256
CLIP_LEN=5
BATCH_SIZE=1
NUM_WORKERS=2

SLOT_STEPS="${SLOT_STEPS:-30000}"
SELECTION_STEPS="${SELECTION_STEPS:-50000}"
CAPACITY_STEPS="${CAPACITY_STEPS:-30000}"
ROBUST_STEPS="${ROBUST_STEPS:-80000}"
JOINT_STEPS="${JOINT_STEPS:-30000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --model-i) MODEL_I="$2"; shift 2 ;;
    --model-p) MODEL_P="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --img-h) IMG_H="$2"; shift 2 ;;
    --img-w) IMG_W="$2"; shift 2 ;;
    --clip-len) CLIP_LEN="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "Usage: $0 --data-root /path/to/frame_dataset [--model-i path] [--model-p path]" >&2
  exit 2
fi

COMMON=(
  --data_dir "$DATA_ROOT"
  --model_path_i "$MODEL_I"
  --model_path_p "$MODEL_P"
  --device "$DEVICE"
  --img_h "$IMG_H"
  --img_w "$IMG_W"
  --clip_len "$CLIP_LEN"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --val_conditions "12,10,0.0;5,10,0.0;12,10,0.2;10,2,0.0;20,18,0.0;5,2,0.1"
)

python -m training.train_dcvcfm_satellite_curriculum \
  "${COMMON[@]}" \
  --phase slot_warmup \
  --channel_type identity \
  --save_dir "$OUT_ROOT/01_slot_warmup" \
  --lr 2e-4 \
  --max_steps "$SLOT_STEPS"

python -m training.train_dcvcfm_satellite_curriculum \
  "${COMMON[@]}" \
  --phase selection_warmup \
  --channel_type identity \
  --resume "$OUT_ROOT/01_slot_warmup/best.pt" \
  --save_dir "$OUT_ROOT/02_selection_warmup" \
  --lr 1e-4 \
  --max_steps "$SELECTION_STEPS"

python -m training.train_dcvcfm_satellite_curriculum \
  "${COMMON[@]}" \
  --phase capacity_calibration \
  --channel_type identity \
  --resume "$OUT_ROOT/02_selection_warmup/best.pt" \
  --save_dir "$OUT_ROOT/03_capacity_calibration" \
  --lr 8e-5 \
  --lambda_monotonic 0.5 \
  --max_steps "$CAPACITY_STEPS"

python -m training.train_dcvcfm_satellite_curriculum \
  "${COMMON[@]}" \
  --phase robust_curriculum \
  --channel_type satellite \
  --resume "$OUT_ROOT/03_capacity_calibration/best.pt" \
  --save_dir "$OUT_ROOT/04_robust_curriculum" \
  --lr 6e-5 \
  --lambda_channel 0.08 \
  --max_steps "$ROBUST_STEPS"

python -m training.train_dcvcfm_satellite_curriculum \
  "${COMMON[@]}" \
  --phase joint_finetune \
  --channel_type satellite \
  --resume "$OUT_ROOT/04_robust_curriculum/best.pt" \
  --save_dir "$OUT_ROOT/05_joint_finetune" \
  --lr 1e-5 \
  --lambda_channel 0.05 \
  --lambda_monotonic 0.25 \
  --max_steps "$JOINT_STEPS"

python -m training.evaluate_dcvcfm_satellite_suite \
  --data_dir "$DATA_ROOT" \
  --ckpt "$OUT_ROOT/05_joint_finetune/best.pt" \
  --model_path_i "$MODEL_I" \
  --model_path_p "$MODEL_P" \
  --device "$DEVICE" \
  --img_h "$IMG_H" \
  --img_w "$IMG_W" \
  --clip_len "$CLIP_LEN" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --channel_type satellite \
  --output_dir "results/dcvcfm_satellite_curriculum"
