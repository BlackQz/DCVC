"""Run the formal satellite DCVC-FM evaluation suite.

The single-condition evaluator remains the source of truth for metrics.  This
suite orchestrates tiers and scans, then writes a compact diagnostic JSON for
bandwidth response and robustness gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_dcvcfm_satellite import evaluate  # noqa: E402
from training.satellite_curriculum import bandwidth_scan_diagnostics, parse_float_list  # noqa: E402


LOGGER = logging.getLogger("evaluate_dcvcfm_satellite_suite")


def _mean(summary: dict[str, Any], section: str, key: str) -> float | None:
    value = summary.get(section, {}).get(key, {}).get("mean")
    return None if value is None else float(value)


def flatten_summary(summary: dict[str, Any], *, label: str, suite: str, snr: float, bw: float, plr: float) -> dict[str, Any]:
    return {
        "suite": suite,
        "label": label,
        "snr_db": float(snr),
        "bandwidth_mbps": float(bw),
        "packet_loss_rate": float(plr),
        "PSNR": _mean(summary, "visual", "PSNR"),
        "SSIM": _mean(summary, "visual", "SSIM"),
        "MS-SSIM": _mean(summary, "visual", "MS-SSIM"),
        "LPIPS": _mean(summary, "visual", "LPIPS"),
        "DISTS": _mean(summary, "visual", "DISTS"),
        "VMAF": _mean(summary, "visual", "VMAF"),
        "bpp": _mean(summary, "rate", "bpp"),
        "kbps": _mean(summary, "rate", "kbps"),
        "target_bpp": _mean(summary, "rate", "target_bpp"),
        "keep_ratio": _mean(summary, "rate", "keep_ratio"),
        "base_layer_ratio": _mean(summary, "rate", "base_layer_ratio"),
        "enhancement_layer_ratio": _mean(summary, "rate", "enhancement_layer_ratio"),
        "proc_time_ms": _mean(summary, "time", "proc_time_ms"),
        "tx_time_ms": _mean(summary, "time", "tx_time_ms"),
        "capacity_mbps": summary.get("channel", {}).get("capacity_mbps", {}).get("mean"),
        "input_source": summary.get("input_source"),
        "num_gops": summary.get("num_gops"),
        "num_frames": summary.get("num_frames"),
    }


def suite_conditions(args: argparse.Namespace) -> list[tuple[str, str, float, float, float]]:
    suites = {item.strip() for item in args.suites.split(",") if item.strip()}
    if "all" in suites:
        suites = {"tiers", "bandwidth", "snr", "plr"}
    conditions: list[tuple[str, str, float, float, float]] = []

    if "tiers" in suites:
        conditions.extend(
            [
                ("tiers", "outage", 1.0, 1.0, 0.50),
                ("tiers", "poor", 5.0, 2.0, 0.30),
                ("tiers", "medium", 10.0, 5.0, 0.10),
                ("tiers", "good", 20.0, 20.0, 0.00),
            ]
        )
    if "bandwidth" in suites:
        for bw in parse_float_list(args.bandwidth_scan):
            conditions.append(("bandwidth", f"bw_{bw:g}", args.bandwidth_scan_snr_db, bw, args.bandwidth_scan_plr))
    if "snr" in suites:
        for snr in parse_float_list(args.snr_scan):
            conditions.append(("snr", f"snr_{snr:g}", snr, args.snr_scan_bw_mbps, args.snr_scan_plr))
    if "plr" in suites:
        for plr in parse_float_list(args.plr_scan):
            conditions.append(("plr", f"plr_{plr:g}", args.plr_scan_snr_db, args.plr_scan_bw_mbps, plr))

    unknown = suites.difference({"tiers", "bandwidth", "snr", "plr"})
    if unknown:
        raise ValueError(f"unknown suites: {', '.join(sorted(unknown))}")
    return conditions


def make_eval_args(
    args: argparse.Namespace,
    *,
    suite: str,
    label: str,
    snr: float,
    bw: float,
    plr: float,
) -> argparse.Namespace:
    eval_args = copy.deepcopy(args)
    eval_args.output_dir = str(Path(args.output_dir) / suite / label)
    eval_args.snr_db = float(snr)
    eval_args.bandwidth_mbps = float(bw)
    eval_args.packet_loss_rate = float(plr)
    return eval_args


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    if args.channel_type == "auto":
        args.channel_type = "satellite"
    conditions = suite_conditions(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary = {"dry_run": True, "conditions": conditions}
        (out_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    flat_results = []
    raw_results = {}
    for suite, label, snr, bw, plr in conditions:
        LOGGER.info("eval suite=%s label=%s SNR=%.1f BW=%.1f PLR=%.2f", suite, label, snr, bw, plr)
        eval_args = make_eval_args(args, suite=suite, label=label, snr=snr, bw=bw, plr=plr)
        summary = evaluate(eval_args)
        condition_dir = Path(eval_args.output_dir)
        condition_dir.mkdir(parents=True, exist_ok=True)
        (condition_dir / "eval_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        flat = flatten_summary(summary, label=label, suite=suite, snr=snr, bw=bw, plr=plr)
        flat_results.append(flat)
        raw_results[f"{suite}/{label}"] = {
            "eval_results": str((condition_dir / "eval_results.json").resolve()),
            "summary": flat,
        }

    diagnostics: dict[str, Any] = {}
    bandwidth_items = [item for item in flat_results if item["suite"] == "bandwidth"]
    if bandwidth_items:
        diagnostics["bandwidth"] = bandwidth_scan_diagnostics(bandwidth_items)
    high_bw = [
        item for item in flat_results
        if item["snr_db"] >= 20.0 and item["bandwidth_mbps"] >= 20.0 and item["packet_loss_rate"] == 0.0
    ]
    if high_bw:
        diagnostics["high_bandwidth_clean"] = {
            "min_PSNR": min(float(item["PSNR"] or 0.0) for item in high_bw),
            "mean_bpp": sum(float(item["bpp"] or 0.0) for item in high_bw) / len(high_bw),
            "conditions": [item["label"] for item in high_bw],
        }
    plr_items = [item for item in flat_results if item["suite"] == "plr"]
    if plr_items:
        by_plr = sorted(plr_items, key=lambda item: item["packet_loss_rate"])
        diagnostics["plr_robustness"] = {
            "plr": [item["packet_loss_rate"] for item in by_plr],
            "PSNR": [item["PSNR"] for item in by_plr],
            "keep_ratio": [item["keep_ratio"] for item in by_plr],
            "has_plr_0_3": any(abs(item["packet_loss_rate"] - 0.3) < 1e-6 for item in by_plr),
            "has_plr_0_5": any(abs(item["packet_loss_rate"] - 0.5) < 1e-6 for item in by_plr),
        }

    ckpt_value = None
    if args.ckpt and str(args.ckpt).lower() not in {"none", "null", "false"}:
        ckpt_value = str(Path(args.ckpt).resolve())
    summary = {
        "checkpoint": ckpt_value,
        "input_source": str(Path(args.data_dir).resolve()),
        "channel_type": args.channel_type,
        "disable_satellite": bool(args.disable_satellite),
        "conditions": flat_results,
        "diagnostics": diagnostics,
        "raw_result_files": raw_results,
    }
    (out_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formal evaluation suite for satellite-aware DCVC-FM")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="checkpoints/dcvcfm_satellite_curriculum/best.pt")
    p.add_argument("--model_path_i", type=str, default="")
    p.add_argument("--model_path_p", type=str, default="")
    p.add_argument("--output_dir", type=str, default="results/dcvcfm_satellite_suite")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--clip_len", type=int, default=5)
    p.add_argument("--clip_stride", type=int, default=4)
    p.add_argument("--max_gops", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)

    p.add_argument("--suites", type=str, default="all", help="comma list: all,tiers,bandwidth,snr,plr")
    p.add_argument("--bandwidth_scan", type=str, default="1,2,5,10,20,25")
    p.add_argument("--bandwidth_scan_snr_db", type=float, default=20.0)
    p.add_argument("--bandwidth_scan_plr", type=float, default=0.0)
    p.add_argument("--snr_scan", type=str, default="1,5,10,15,20")
    p.add_argument("--snr_scan_bw_mbps", type=float, default=10.0)
    p.add_argument("--snr_scan_plr", type=float, default=0.0)
    p.add_argument("--plr_scan", type=str, default="0,0.05,0.1,0.2,0.3,0.5")
    p.add_argument("--plr_scan_snr_db", type=float, default=12.0)
    p.add_argument("--plr_scan_bw_mbps", type=float, default=10.0)
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--channel_type", type=str, default="auto", choices=["auto", "identity", "awgn", "rayleigh", "satellite"])
    p.add_argument("--disable_satellite", action="store_true")
    p.add_argument("--disable_lpips", action="store_true")
    p.add_argument("--disable_dists", action="store_true")
    p.add_argument("--num_slots", type=int, default=7)
    p.add_argument("--slot_dim", type=int, default=64)
    p.add_argument("--slot_output_dim", type=int, default=128)
    p.add_argument("--slot_iterations", type=int, default=3)
    p.add_argument("--slot_adapter_h", type=int, default=128)
    p.add_argument("--slot_adapter_w", type=int, default=128)
    p.add_argument("--no_update_slots_on_p", action="store_true")

    p.add_argument("--q_index_i", type=int, default=63)
    p.add_argument("--q_index_p", type=int, default=63)
    p.set_defaults(capacity_q_index=True)
    p.add_argument("--fixed_q_index", dest="capacity_q_index", action="store_false")
    p.add_argument("--intra_period", type=int, default=9999)

    p.add_argument("--min_capacity_mbps", type=float, default=0.5)
    p.add_argument("--max_capacity_mbps", type=float, default=110.0)
    p.add_argument("--min_target_bpp", type=float, default=0.035)
    p.add_argument("--max_target_bpp", type=float, default=0.260)
    p.add_argument("--target_tolerance_low", type=float, default=0.18)
    p.add_argument("--target_tolerance_high", type=float, default=0.25)
    p.add_argument("--min_keep_ratio", type=float, default=0.16)
    p.add_argument("--max_keep_ratio", type=float, default=0.92)
    p.add_argument("--base_min_keep_ratio", type=float, default=0.16)
    p.add_argument("--base_max_keep_ratio", type=float, default=0.46)
    p.add_argument("--enhancement_start_norm", type=float, default=0.28)
    p.add_argument("--keep_gamma", type=float, default=0.85)
    p.add_argument("--q_index_low_capacity", type=float, default=0.0)
    p.add_argument("--q_index_high_capacity", type=float, default=63.0)
    p.add_argument("--ste_temperature", type=float, default=0.08)
    p.add_argument("--disable_learnable_capacity_offsets", action="store_true")

    p.add_argument("--no_row_packet_loss", action="store_true")
    p.add_argument("--base_snr_gain_db", type=float, default=6.0)
    p.add_argument("--enhancement_snr_gain_db", type=float, default=0.0)
    p.add_argument("--base_plr_scale", type=float, default=0.25)
    p.add_argument("--enhancement_plr_scale", type=float, default=1.25)
    p.add_argument("--rician_k_db", type=float, default=10.0)
    p.add_argument("--shadowing_std_db", type=float, default=3.0)
    p.add_argument("--beam_switch_prob", type=float, default=0.01)
    args = p.parse_args()
    parse_float_list(args.bandwidth_scan)
    parse_float_list(args.snr_scan)
    parse_float_list(args.plr_scan)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_suite(parse_args())
    LOGGER.info("wrote suite summary with %d conditions", len(summary.get("conditions", [])))


if __name__ == "__main__":
    main()
