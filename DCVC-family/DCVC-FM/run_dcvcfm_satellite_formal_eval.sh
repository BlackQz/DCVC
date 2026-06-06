#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${DATA_ROOT:-}"
CHECKPOINT="${CHECKPOINT:-checkpoints/dcvcfm_satellite/best.pt}"
MODEL_PATH_I="${MODEL_PATH_I:-checkpoints/cvpr2024_image.pth.tar}"
MODEL_PATH_P="${MODEL_PATH_P:-checkpoints/cvpr2024_video.pth.tar}"
PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
RESULTS_ROOT="${RESULTS_ROOT:-results/dcvcfm_satellite}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_GOPS="${MAX_GOPS:-0}"
CONTINUE_ON_ERROR=0

usage() {
    cat <<'EOF'
Run formal satellite-aware DCVC-FM evaluation.

Usage:
  bash run_dcvcfm_satellite_formal_eval.sh --data-root /path/to/frame/folders

Options:
  --data-root PATH       Dataset root. Can also be set with DATA_ROOT.
  --checkpoint PATH      Satellite checkpoint. Default: checkpoints/dcvcfm_satellite/best.pt
  --model-path-i PATH    DCVC-FM intra checkpoint.
  --model-path-p PATH    DCVC-FM video checkpoint.
  --results-root PATH    Results root. Default: results/dcvcfm_satellite
  --device DEVICE        cuda or cpu. Default: cuda
  --num-workers N        DataLoader workers. Default: 2
  --max-gops N           Limit evaluated GoPs. Default: 0 means all.
  --python CMD           Python command. Default: python, or PYTHON env var.
  --continue-on-error    Continue remaining scans after a failed run.
  -h, --help             Show help.

Scans:
  tier: outage / poor / medium / good
  bandwidth: 1 / 2 / 5 / 10 / 20 / 25 Mbps
  SNR: 1 / 5 / 10 / 15 / 20 dB
  PLR: 0 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --model-path-i) MODEL_PATH_I="$2"; shift 2 ;;
        --model-path-p) MODEL_PATH_P="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --max-gops) MAX_GOPS="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ -z "$DATA_ROOT" || ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: valid --data-root is required." >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

failed_runs=()

run_eval() {
    local name="$1"
    local snr="$2"
    local bw="$3"
    local plr="$4"
    local out_dir="$5"
    local cmd=(
        "$PYTHON_BIN" -m training.evaluate_dcvcfm_satellite
        --data_dir "$DATA_ROOT"
        --ckpt "$CHECKPOINT"
        --model_path_i "$MODEL_PATH_I"
        --model_path_p "$MODEL_PATH_P"
        --channel_type satellite
        --snr_db "$snr"
        --bandwidth_mbps "$bw"
        --packet_loss_rate "$plr"
        --output_dir "$out_dir"
        --device "$DEVICE"
        --num_workers "$NUM_WORKERS"
    )
    if [[ "$MAX_GOPS" -gt 0 ]]; then
        cmd+=(--max_gops "$MAX_GOPS")
    fi
    echo
    echo "Running $name: SNR=$snr BW=$bw PLR=$plr"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if ! "${cmd[@]}"; then
        if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
            failed_runs+=("$name")
            return 0
        fi
        exit 1
    fi
}

run_eval "tier/outage" 2 2 0.15 "$RESULTS_ROOT/tier/outage"
run_eval "tier/poor" 6 5 0.08 "$RESULTS_ROOT/tier/poor"
run_eval "tier/medium" 12 10 0.03 "$RESULTS_ROOT/tier/medium"
run_eval "tier/good" 22 20 0.00 "$RESULTS_ROOT/tier/good"

for bw in 1 2 5 10 20 25; do
    run_eval "bandwidth/${bw}Mbps" 10 "$bw" 0.0 "$RESULTS_ROOT/bandwidth/bw_${bw}"
done

for snr in 1 5 10 15 20; do
    run_eval "snr/${snr}dB" "$snr" 10 0.0 "$RESULTS_ROOT/snr/snr_${snr}"
done

for plr in 0 0.05 0.1 0.2 0.3 0.5; do
    run_eval "plr/${plr}" 10 10 "$plr" "$RESULTS_ROOT/plr/plr_${plr}"
done

if [[ "${#failed_runs[@]}" -gt 0 ]]; then
    echo "Failed runs:" >&2
    printf '  - %s\n' "${failed_runs[@]}" >&2
    exit 1
fi

echo "Formal satellite evaluation completed."
