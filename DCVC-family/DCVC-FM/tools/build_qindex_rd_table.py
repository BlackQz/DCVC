"""Build a lightweight q-index RD table for CSI-aware DCVC-FM control.

The table is an offline calibration artifact, not a training loop.  By default
it evaluates only nine anchor q points and lets the controller interpolate
between them, avoiding a 64x training cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satellite import SatelliteDCVCFM  # noqa: E402
from training.train_dcvcfm_satellite import VideoFolderDataset, compute_ssim_torch  # noqa: E402


def parse_q_indexes(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        q = int(item)
        if q < 0 or q > 63:
            raise ValueError(f"q_index must be in [0, 63], got {q}")
        values.append(q)
    if not values:
        raise ValueError("at least one q_index is required")
    return sorted(set(values))


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).detach().to(torch.float32).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate_q_point(model: SatelliteDCVCFM, loader: DataLoader, *, q_index: int, args: argparse.Namespace, device: torch.device) -> dict:
    bpps = []
    psnrs = []
    ssims = []
    frames = 0
    begin = time.perf_counter()
    for batch_idx, clip in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        clip = clip.to(device, non_blocking=True)
        out = model.forward_sequence(
            clip,
            snr_db=args.snr_db,
            bandwidth_mbps=args.bandwidth_mbps,
            packet_loss_rate=args.packet_loss_rate,
            q_index_i=q_index,
            q_index_p=q_index,
            intra_period=args.intra_period,
            enable_satellite=False,
        )
        recon = out["x_hat"].clamp(0.0, 1.0)
        bpps.append(float(out["bpp"].detach().cpu().item()))
        psnrs.append(compute_psnr(recon, clip))
        ssims.append(compute_ssim_torch(recon, clip))
        frames += int(clip.shape[0] * clip.shape[1])
    count = max(len(bpps), 1)
    return {
        "q_index": int(q_index),
        "bpp": float(sum(bpps) / count),
        "PSNR": float(sum(psnrs) / count),
        "SSIM": float(sum(ssims) / count),
        "num_gops": int(len(bpps)),
        "num_frames": int(frames),
        "elapsed_sec": float(time.perf_counter() - begin),
    }


def build_table(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    dataset = VideoFolderDataset(
        data_root,
        clip_len=args.clip_len,
        image_size=(args.img_h, args.img_w),
        stride=args.clip_stride,
        max_clips=args.max_clips,
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = SatelliteDCVCFM.from_pretrained(
        model_path_i=args.model_path_i or None,
        model_path_p=args.model_path_p or None,
        strict=not args.non_strict_load,
        channel_type="identity",
        update_slots_on_p=False,
    ).to(device).eval()

    q_points = []
    for q in parse_q_indexes(args.q_indexes):
        item = evaluate_q_point(model, loader, q_index=q, args=args, device=device)
        print(
            f"q={q:02d} bpp={item['bpp']:.5f} PSNR={item['PSNR']:.2f} "
            f"SSIM={item['SSIM']:.4f} gops={item['num_gops']}"
        )
        q_points.append(item)

    table = {
        "format": "dcvcfm_satellite_qindex_rd_table_v1",
        "description": "Anchor q-index RD table for CSI-aware controller interpolation.",
        "data_dir": str(data_root.resolve()),
        "model_path_i": args.model_path_i,
        "model_path_p": args.model_path_p,
        "clip_len": args.clip_len,
        "image_size": [args.img_h, args.img_w],
        "max_clips": args.max_clips,
        "max_batches": args.max_batches,
        "q_points": q_points,
        "note": "Training still executes one q_index per sample; this table avoids evaluating all 64 q points inside training.",
    }
    return table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build q-index RD table for satellite-aware DCVC-FM")
    p.add_argument("--data_dir", type=str, required=True, help="validation frame folder, e.g. BVI-DVC-512/val")
    p.add_argument("--model_path_i", type=str, required=True)
    p.add_argument("--model_path_p", type=str, required=True)
    p.add_argument("--output", type=str, default="results/qindex_rd_table_bvidvc512.json")
    p.add_argument("--q_indexes", type=str, default="0,8,16,24,32,40,48,56,63")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=512)
    p.add_argument("--clip_len", type=int, default=7)
    p.add_argument("--clip_stride", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_clips", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--intra_period", type=int, default=9999)
    p.add_argument("--snr_db", type=float, default=20.0)
    p.add_argument("--bandwidth_mbps", type=float, default=25.0)
    p.add_argument("--packet_loss_rate", type=float, default=0.0)
    p.add_argument("--non_strict_load", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
