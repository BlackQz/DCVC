#!/usr/bin/env bash
# 卫星感知 DCVC-FM 主训练脚本（Ubuntu / 单机最多 8 卡）
#
# 设计来源：
# - DCVC-FM 官方测试方式：保留官方 I/P 预训练权重、变码率 q_index 覆盖、多 GPU worker 思路；
# - DCVC-FM 训练经验：多帧级联、周期 intra、后期小学习率微调；
# - Slot Attention 官方训练方式：128x128 object discovery、L2 重建、Adam、warmup + exponential decay；
# - 当前卫星模型需求：连续容量控制、base/enhancement latent、SNR/BW/PLR 鲁棒性。
#
# 推荐用法：
#   cd DCVC-family/DCVC-FM
#   bash run_dcvcfm_satellite_curriculum_8gpu.sh --data-root /data/sdb/bitqzh/data/BVI-DVC-512 --ngpu 8

set -euo pipefail

DATA_ROOT=""
MODEL_I="checkpoints/cvpr2024_image.pth.tar"
MODEL_P="checkpoints/cvpr2024_video.pth.tar"
OUT_ROOT="checkpoints/dcvcfm_satellite_curriculum"
RESULT_ROOT="results/dcvcfm_satellite_curriculum"
TRAIN_IMG_H=256
TRAIN_IMG_W=256
EVAL_IMG_H=512
EVAL_IMG_W=512
BATCH_SIZE=1
NUM_WORKERS=6
NGPU="${NGPU:-8}"

# 这些是“单卡等效总步数”。脚本会自动除以 NGPU，让 8 卡看到的总样本量接近单卡设置。
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
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --train-img-h) TRAIN_IMG_H="$2"; shift 2 ;;
    --train-img-w) TRAIN_IMG_W="$2"; shift 2 ;;
    --eval-img-h) EVAL_IMG_H="$2"; shift 2 ;;
    --eval-img-w) EVAL_IMG_W="$2"; shift 2 ;;
    --img-h) TRAIN_IMG_H="$2"; EVAL_IMG_H="$2"; shift 2 ;;
    --img-w) TRAIN_IMG_W="$2"; EVAL_IMG_W="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --ngpu) NGPU="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "用法: $0 --data-root /path/to/frame_dataset [--ngpu 8]" >&2
  exit 2
fi

if [[ "$NGPU" -lt 1 || "$NGPU" -gt 8 ]]; then
  echo "NGPU 必须在 1 到 8 之间，当前为 $NGPU" >&2
  exit 2
fi

div_steps() {
  local total="$1"
  echo $(( (total + NGPU - 1) / NGPU ))
}

scale_steps() {
  local total="$1"
  echo $(( (total + NGPU - 1) / NGPU ))
}

SLOT_S=$(div_steps "$SLOT_STEPS")
SEL_S=$(div_steps "$SELECTION_STEPS")
CAP_S=$(div_steps "$CAPACITY_STEPS")
ROB_S=$(div_steps "$ROBUST_STEPS")
JOINT_S=$(div_steps "$JOINT_STEPS")

SLOT_WARMUP=$(scale_steps 10000)
SLOT_DECAY=$(scale_steps 100000)
ADAPTER_WARMUP=$(scale_steps 5000)
ADAPTER_DECAY=$(scale_steps 60000)

echo "[DCVC-FM-SAT] NGPU=$NGPU per-rank steps: slot=$SLOT_S selection=$SEL_S capacity=$CAP_S robust=$ROB_S joint=$JOINT_S"
echo "[DCVC-FM-SAT] checkpoints: $OUT_ROOT"
echo "[DCVC-FM-SAT] train crop: ${TRAIN_IMG_H}x${TRAIN_IMG_W}; eval frame: ${EVAL_IMG_H}x${EVAL_IMG_W}"
if [[ -d "$DATA_ROOT/val" ]]; then
  EVAL_DATA_DIR="$DATA_ROOT/val"
elif [[ -d "$DATA_ROOT/test" ]]; then
  EVAL_DATA_DIR="$DATA_ROOT/test"
else
  EVAL_DATA_DIR="$DATA_ROOT"
fi
echo "[DCVC-FM-SAT] eval data: $EVAL_DATA_DIR"

if [[ "$NGPU" -eq 1 ]]; then
  LAUNCH=(python -m training.train_dcvcfm_satellite_curriculum)
else
  LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" -m training.train_dcvcfm_satellite_curriculum)
fi

COMMON=(
  --data_dir "$DATA_ROOT"
  --model_path_i "$MODEL_I"
  --model_path_p "$MODEL_P"
  --device cuda
  --img_h "$TRAIN_IMG_H"
  --img_w "$TRAIN_IMG_W"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --val_conditions "12,10,0.0;5,10,0.0;12,10,0.2;10,2,0.0;20,18,0.0;5,2,0.1"
)

# 0. 纯 DCVC-FM wrapper 基线。只验证路径，不训练。
"${LAUNCH[@]}" "${COMMON[@]}" \
  --phase baseline \
  --channel_type identity \
  --save_dir "$OUT_ROOT/00_baseline" \
  --clip_len 5 \
  --max_steps 0

# 1. Slot Attention 预热。对齐官方 object discovery：128x128、L2 重建、warmup + decay。
"${LAUNCH[@]}" "${COMMON[@]}" \
  --phase slot_warmup \
  --channel_type identity \
  --save_dir "$OUT_ROOT/01_slot_warmup" \
  --clip_len 4 \
  --slot_adapter_h 128 --slot_adapter_w 128 \
  --lr 2e-4 \
  --lr_warmup_steps "$SLOT_WARMUP" \
  --lr_decay_steps "$SLOT_DECAY" \
  --lr_decay_rate 0.5 \
  --max_steps "$SLOT_S"

# 2. 语义选择预热。冻结 DCVC-FM，identity channel，先把 selector/capacity 学活。
"${LAUNCH[@]}" "${COMMON[@]}" \
  --phase selection_warmup \
  --channel_type identity \
  --resume "$OUT_ROOT/01_slot_warmup/best.pt" \
  --save_dir "$OUT_ROOT/02_selection_warmup" \
  --clip_len 5 \
  --lr 1e-4 \
  --lr_warmup_steps "$ADAPTER_WARMUP" \
  --lr_decay_steps "$ADAPTER_DECAY" \
  --lr_decay_rate 0.7 \
  --max_steps "$SEL_S"

# 3. 连续容量校准。成对带宽 + 单调性损失，直接解决“所有带宽都省码率”。
"${LAUNCH[@]}" "${COMMON[@]}" \
  --phase capacity_calibration \
  --channel_type identity \
  --resume "$OUT_ROOT/02_selection_warmup/best.pt" \
  --save_dir "$OUT_ROOT/03_capacity_calibration" \
  --clip_len 5 \
  --lr 8e-5 \
  --lr_warmup_steps "$ADAPTER_WARMUP" \
  --lr_decay_steps "$ADAPTER_DECAY" \
  --lr_decay_rate 0.7 \
  --lambda_monotonic 0.5 \
  --max_steps "$CAP_S"

# 4. 卫星信道鲁棒课程。引入周期 intra 和更长 clip，贴近 DCVC-FM 多帧级联训练。
"${LAUNCH[@]}" "${COMMON[@]}" \
  --phase robust_curriculum \
  --channel_type satellite \
  --resume "$OUT_ROOT/03_capacity_calibration/best.pt" \
  --save_dir "$OUT_ROOT/04_robust_curriculum" \
  --clip_len 7 \
  --intra_period 8 \
  --lr 6e-5 \
  --lr_warmup_steps "$ADAPTER_WARMUP" \
  --lr_decay_steps "$ADAPTER_DECAY" \
  --lr_decay_rate 0.7 \
  --lambda_channel 0.08 \
  --max_steps "$ROB_S"

# 5. 小学习率联合微调。只解冻 DCVC-FM 的后端调制/先验融合/decoder 相关模块。
"${LAUNCH[@]}" "${COMMON[@]}" \
  --phase joint_finetune \
  --channel_type satellite \
  --resume "$OUT_ROOT/04_robust_curriculum/best.pt" \
  --save_dir "$OUT_ROOT/05_joint_finetune" \
  --clip_len 7 \
  --intra_period 8 \
  --lr 1e-5 \
  --lr_warmup_steps "$ADAPTER_WARMUP" \
  --lr_decay_steps "$ADAPTER_DECAY" \
  --lr_decay_rate 0.8 \
  --lambda_channel 0.05 \
  --lambda_monotonic 0.25 \
  --max_steps "$JOINT_S"

# 正式评估默认单卡即可。训练如需 8 卡，评估更适合用单卡固定条件逐项扫。
CUDA_VISIBLE_DEVICES=0 python -m training.evaluate_dcvcfm_satellite_suite \
  --data_dir "$EVAL_DATA_DIR" \
  --ckpt "$OUT_ROOT/05_joint_finetune/best.pt" \
  --model_path_i "$MODEL_I" \
  --model_path_p "$MODEL_P" \
  --device cuda \
  --img_h "$EVAL_IMG_H" \
  --img_w "$EVAL_IMG_W" \
  --clip_len 7 \
  --batch_size 1 \
  --num_workers "$NUM_WORKERS" \
  --channel_type satellite \
  --output_dir "$RESULT_ROOT"

echo "[DCVC-FM-SAT] 全部阶段完成。正式 checkpoint: $OUT_ROOT/05_joint_finetune/best.pt"
