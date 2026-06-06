"""Evaluate satellite-aware DCVC-FM under one channel condition."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satellite import SatelliteDCVCFM  # noqa: E402
try:
    from training.train_dcvcfm_satellite import VideoFolderDataset, compute_ssim_torch  # noqa: E402
except ModuleNotFoundError:
    from train_dcvcfm_satellite import VideoFolderDataset, compute_ssim_torch  # type: ignore  # noqa: E402


LOGGER = logging.getLogger("evaluate_dcvcfm_satellite")


def metric_stats(values: list[Optional[float]]) -> dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).detach().to(torch.float32).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ms_ssim_fallback(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_f = pred.flatten(0, 1)
    target_f = target.flatten(0, 1)
    weights = [0.40, 0.30, 0.20, 0.10]
    total = 0.0
    for idx, weight in enumerate(weights):
        total += weight * compute_ssim_4d(pred_f, target_f)
        if idx != len(weights) - 1 and min(pred_f.shape[-2:]) >= 32:
            pred_f = F.avg_pool2d(pred_f, 2)
            target_f = F.avg_pool2d(target_f, 2)
    return float(total)


def compute_ssim_4d(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
    sig_x = F.avg_pool2d(pred.square(), 11, 1, 5) - mu_x.square()
    sig_y = F.avg_pool2d(target.square(), 11, 1, 5) - mu_y.square()
    sig_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sig_x + sig_y + c2)
    )
    return float(ssim.mean().detach().cpu().item())


class OptionalLPIPS:
    def __init__(self, device: torch.device, disabled: bool = False) -> None:
        self.device = device
        self.disabled = disabled
        self.model = None
        self.failed = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
        if self.disabled or self.failed:
            return None
        if self.model is None:
            try:
                import lpips
                self.model = lpips.LPIPS(net="alex").to(self.device).eval()
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("LPIPS unavailable: %s", exc)
                self.failed = True
                return None
        with torch.no_grad():
            return float(self.model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().cpu().item())


class OptionalDISTS:
    def __init__(self, device: torch.device, disabled: bool = False) -> None:
        self.device = device
        self.disabled = disabled
        self.model = None
        self.failed = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
        if self.disabled or self.failed:
            return None
        if self.model is None:
            try:
                from DISTS_pytorch import DISTS
                self.model = DISTS().to(self.device).eval()
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("DISTS unavailable: %s", exc)
                self.failed = True
                return None
        with torch.no_grad():
            return float(self.model(pred, target).mean().cpu().item())


def load_model(args: argparse.Namespace, device: torch.device) -> SatelliteDCVCFM:
    model = SatelliteDCVCFM.from_pretrained(
        model_path_i=args.model_path_i or None,
        model_path_p=args.model_path_p or None,
        strict=False,
        channel_type=args.channel_type,
        slot_num=args.num_slots,
        slot_dim=args.slot_dim,
        slot_output_dim=args.slot_output_dim,
        slot_iterations=args.slot_iterations,
        slot_adapter_resolution=(args.slot_adapter_h, args.slot_adapter_w),
        update_slots_on_p=not args.no_update_slots_on_p,
        controller_kwargs={
            "min_capacity_mbps": args.min_capacity_mbps,
            "max_capacity_mbps": args.max_capacity_mbps,
            "min_target_bpp": args.min_target_bpp,
            "max_target_bpp": args.max_target_bpp,
            "target_tolerance_low": args.target_tolerance_low,
            "target_tolerance_high": args.target_tolerance_high,
            "min_keep_ratio": args.min_keep_ratio,
            "max_keep_ratio": args.max_keep_ratio,
            "base_min_keep_ratio": args.base_min_keep_ratio,
            "base_max_keep_ratio": args.base_max_keep_ratio,
            "enhancement_start_norm": args.enhancement_start_norm,
            "keep_gamma": args.keep_gamma,
            "q_index_low_capacity": args.q_index_low_capacity,
            "q_index_high_capacity": args.q_index_high_capacity,
            "learnable_offsets": not args.disable_learnable_capacity_offsets,
        },
        selector_kwargs={
            "ste_temperature": args.ste_temperature,
        },
        channel_kwargs={
            "row_packet_loss": not args.no_row_packet_loss,
            "base_snr_gain_db": args.base_snr_gain_db,
            "enhancement_snr_gain_db": args.enhancement_snr_gain_db,
            "base_plr_scale": args.base_plr_scale,
            "enhancement_plr_scale": args.enhancement_plr_scale,
            "rician_k_db": args.rician_k_db,
            "shadowing_std_db": args.shadowing_std_db,
            "beam_switch_prob": args.beam_switch_prob,
        },
    ).to(device)
    if args.ckpt and str(args.ckpt).lower() not in {"none", "null", "false"}:
        payload = torch.load(args.ckpt, map_location=device)
        state = payload.get("model_state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        LOGGER.info("checkpoint loaded: %s | missing=%d unexpected=%d", args.ckpt, len(missing), len(unexpected))
    return model.eval()


def flatten_frame_metrics(frame_outputs, key: str) -> list[float]:
    return [float(out.metrics[key]) for out in frame_outputs if key in out.metrics]


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values).item()) if values else 0.0


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args, device)
    dataset = VideoFolderDataset(
        args.data_dir,
        clip_len=args.clip_len,
        image_size=(args.img_h, args.img_w),
        stride=args.clip_stride,
        max_clips=args.max_gops,
        random_crop=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    lpips_metric = OptionalLPIPS(device, disabled=args.disable_lpips)
    dists_metric = OptionalDISTS(device, disabled=args.disable_dists)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    msssim_values: list[float] = []
    lpips_values: list[Optional[float]] = []
    dists_values: list[Optional[float]] = []
    bpp_values: list[float] = []
    kbps_values: list[float] = []
    keep_values: list[float] = []
    base_values: list[float] = []
    enhancement_values: list[float] = []
    proc_values: list[float] = []
    tx_values: list[float] = []
    target_bpp_values: list[float] = []
    capacity_values: list[float] = []
    per_gop = []
    total_frames = 0

    start = time.perf_counter()
    for gop_idx, clip in enumerate(loader):
        clip = clip.to(device, non_blocking=True)
        out = model.forward_sequence(
            clip,
            snr_db=args.snr_db,
            bandwidth_mbps=args.bandwidth_mbps,
            packet_loss_rate=args.packet_loss_rate,
            q_index_i=args.q_index_i,
            q_index_p=None if args.capacity_q_index else args.q_index_p,
            intra_period=args.intra_period,
            enable_satellite=not args.disable_satellite,
        )
        recon = out["x_hat"].clamp(0, 1)
        psnr = compute_psnr(recon, clip)
        ssim = compute_ssim_torch(recon, clip)
        msssim = compute_ms_ssim_fallback(recon, clip)
        lp = lpips_metric(recon.flatten(0, 1), clip.flatten(0, 1))
        di = dists_metric(recon.flatten(0, 1), clip.flatten(0, 1))
        pixel_num = clip.shape[-2] * clip.shape[-1]
        bpp = float(out["bpp"].detach().cpu().item())
        kbps = bpp * pixel_num * args.fps / 1000.0
        frames = out["frames"]
        gop_metrics = {
            "gop_index": gop_idx,
            "PSNR": psnr,
            "SSIM": ssim,
            "MS-SSIM": msssim,
            "LPIPS": lp,
            "DISTS": di,
            "VMAF": None,
            "bpp": bpp,
            "kbps": kbps,
            "keep_ratio": safe_mean(flatten_frame_metrics(frames, "keep_ratio")),
            "base_layer_ratio": safe_mean(flatten_frame_metrics(frames, "base_layer_ratio")),
            "enhancement_layer_ratio": safe_mean(flatten_frame_metrics(frames, "enhancement_layer_ratio")),
            "target_bpp": safe_mean(flatten_frame_metrics(frames, "target_bpp")),
            "capacity_mbps": safe_mean(flatten_frame_metrics(frames, "capacity_mbps")),
            "proc_time_ms": safe_mean(flatten_frame_metrics(frames, "proc_time_ms")),
            "tx_time_ms": safe_mean(flatten_frame_metrics(frames, "tx_time_ms")),
            "num_frames": int(clip.shape[1] * clip.shape[0]),
        }
        per_gop.append(gop_metrics)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        msssim_values.append(msssim)
        lpips_values.append(lp)
        dists_values.append(di)
        bpp_values.append(bpp)
        kbps_values.append(kbps)
        keep_values.append(gop_metrics["keep_ratio"])
        base_values.append(gop_metrics["base_layer_ratio"])
        enhancement_values.append(gop_metrics["enhancement_layer_ratio"])
        target_bpp_values.append(gop_metrics["target_bpp"])
        capacity_values.append(gop_metrics["capacity_mbps"])
        proc_values.append(gop_metrics["proc_time_ms"])
        tx_values.append(gop_metrics["tx_time_ms"])
        total_frames += gop_metrics["num_frames"]

        LOGGER.info(
            "GoP %d PSNR=%.2f SSIM=%.4f bpp=%.4f keep=%.3f base=%.3f enh=%.3f",
            gop_idx,
            psnr,
            ssim,
            bpp,
            gop_metrics["keep_ratio"],
            gop_metrics["base_layer_ratio"],
            gop_metrics["enhancement_layer_ratio"],
        )

    elapsed = time.perf_counter() - start
    summary = {
        "input_source": str(Path(args.data_dir).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()) if args.ckpt and str(args.ckpt).lower() not in {"none", "null", "false"} else None,
        "channel": {
            "channel_type": args.channel_type,
            "snr_db": args.snr_db,
            "bandwidth_mbps": args.bandwidth_mbps,
            "packet_loss_rate": args.packet_loss_rate,
            "capacity_mbps": metric_stats(capacity_values),
        },
        "num_gops": len(per_gop),
        "num_frames": total_frames,
        "elapsed_sec": elapsed,
        "visual": {
            "PSNR": metric_stats(psnr_values),
            "SSIM": metric_stats(ssim_values),
            "MS-SSIM": metric_stats(msssim_values),
            "LPIPS": metric_stats(lpips_values),
            "DISTS": metric_stats(dists_values),
            "VMAF": metric_stats([None for _ in per_gop]),
        },
        "rate": {
            "bpp": metric_stats(bpp_values),
            "kbps": metric_stats(kbps_values),
            "target_bpp": metric_stats(target_bpp_values),
            "keep_ratio": metric_stats(keep_values),
            "base_layer_ratio": metric_stats(base_values),
            "enhancement_layer_ratio": metric_stats(enhancement_values),
        },
        "time": {
            "proc_time_ms": metric_stats(proc_values),
            "tx_time_ms": metric_stats(tx_values),
        },
        "per_gop": per_gop,
    }
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = evaluate(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", out_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate satellite-aware DCVC-FM")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="checkpoints/dcvcfm_satellite/best.pt")
    p.add_argument("--model_path_i", type=str, default="")
    p.add_argument("--model_path_p", type=str, default="")
    p.add_argument("--output_dir", type=str, default="results/dcvcfm_satellite_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--clip_len", type=int, default=5)
    p.add_argument("--clip_stride", type=int, default=4)
    p.add_argument("--max_gops", type=int, default=0)
    p.add_argument("--channel_type", type=str, default="satellite", choices=["identity", "awgn", "rayleigh", "satellite"])
    p.add_argument("--snr_db", type=float, default=10.0)
    p.add_argument("--bandwidth_mbps", type=float, default=10.0)
    p.add_argument("--packet_loss_rate", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num_slots", type=int, default=7)
    p.add_argument("--slot_dim", type=int, default=64)
    p.add_argument("--slot_output_dim", type=int, default=128)
    p.add_argument("--slot_iterations", type=int, default=3)
    p.add_argument("--slot_adapter_h", type=int, default=128)
    p.add_argument("--slot_adapter_w", type=int, default=128)
    p.add_argument("--no_update_slots_on_p", action="store_true")
    p.add_argument("--q_index_i", type=int, default=63)
    p.add_argument("--q_index_p", type=int, default=63)
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
    p.set_defaults(capacity_q_index=True)
    p.add_argument("--fixed_q_index", dest="capacity_q_index", action="store_false")
    p.add_argument("--intra_period", type=int, default=9999)
    p.add_argument("--disable_satellite", action="store_true")
    p.add_argument("--disable_lpips", action="store_true")
    p.add_argument("--disable_dists", action="store_true")
    p.add_argument("--no_row_packet_loss", action="store_true")
    p.add_argument("--base_snr_gain_db", type=float, default=6.0)
    p.add_argument("--enhancement_snr_gain_db", type=float, default=0.0)
    p.add_argument("--base_plr_scale", type=float, default=0.25)
    p.add_argument("--enhancement_plr_scale", type=float, default=1.25)
    p.add_argument("--rician_k_db", type=float, default=10.0)
    p.add_argument("--shadowing_std_db", type=float, default=3.0)
    p.add_argument("--beam_switch_prob", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    main()
