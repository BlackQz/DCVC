"""Train satellite-aware adapters on top of DCVC-FM.

Stages:
  A: reproduce/evaluate the original DCVC-FM path without satellite adapters.
  B: freeze DCVC-FM and train Slot/Token/Capacity/Channel adapters.
  C: small-LR joint fine-tuning of feature modulation, quant scalers, and late
     decoder layers.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any
import uuid

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torchvision.io import read_image as _torchvision_read_image
    from torchvision.transforms import functional as _torchvision_tf
    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - optional dependency fallback
    _torchvision_read_image = None
    _torchvision_tf = None
    _HAS_TORCHVISION = False
    from PIL import Image
    import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.satellite import SatelliteDCVCFM, q_index_proxy_loss, rate_budget_interval_loss  # noqa: E402
from src.models.satellite.losses import (  # noqa: E402
    reconstruction_distortion,
    slot_reconstruction_loss,
    temporal_consistency_loss,
    token_selection_regularization,
)


LOGGER = logging.getLogger("train_dcvcfm_satellite")


DEFAULT_VAL_CONDITIONS = "12,10,0.0;5,10,0.0;12,10,0.2;10,2,0.0;20,18,0.0;5,2,0.1"


def read_image_rgb(path: str | Path) -> torch.Tensor:
    if _HAS_TORCHVISION:
        img = _torchvision_read_image(str(path))
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        return _torchvision_tf.convert_image_dtype(img[:3], torch.float32)
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_image(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if _HAS_TORCHVISION:
        return _torchvision_tf.resize(img, list(size), antialias=True)
    return F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def crop_image(img: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    if _HAS_TORCHVISION:
        return _torchvision_tf.crop(img, top, left, height, width)
    return img[:, top:top + height, left:left + width]


class VideoFolderDataset(Dataset):
    EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        root: str | Path,
        *,
        clip_len: int,
        image_size: tuple[int, int],
        stride: int | None = None,
        max_clips: int = 0,
        random_crop: bool = True,
    ) -> None:
        self.root = Path(root)
        self.clip_len = int(clip_len)
        self.image_size = image_size
        self.random_crop = bool(random_crop)
        stride = int(stride or max(1, clip_len - 1))
        frames_by_dir: dict[Path, list[Path]] = {}
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() in self.EXTS:
                frames_by_dir.setdefault(path.parent, []).append(path)

        self.clips: list[list[Path]] = []
        for _, frames in sorted(frames_by_dir.items(), key=lambda item: str(item[0])):
            if len(frames) < 2:
                continue
            if len(frames) < clip_len:
                clip = frames + [frames[-1]] * (clip_len - len(frames))
                self.clips.append(clip)
            else:
                for start in range(0, len(frames) - clip_len + 1, stride):
                    self.clips.append(frames[start:start + clip_len])
        if max_clips > 0:
            self.clips = self.clips[:max_clips]
        if not self.clips:
            raise FileNotFoundError(f"no clips with at least 2 frames found under {self.root}")

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frame(self, path: Path, crop: tuple[int, int, int, int] | None) -> torch.Tensor:
        img = read_image_rgb(path)
        h, w = img.shape[-2:]
        target_h, target_w = self.image_size
        scale = max(target_h / max(h, 1), target_w / max(w, 1), 1.0)
        if scale > 1.0:
            img = resize_image(img, (int(round(h * scale)), int(round(w * scale))))
            h, w = img.shape[-2:]
        if crop is None:
            top = max((h - target_h) // 2, 0)
            left = max((w - target_w) // 2, 0)
        else:
            top, left, _, _ = crop
        return crop_image(img, top, left, target_h, target_w)

    def __getitem__(self, idx: int) -> torch.Tensor:
        paths = self.clips[idx]
        first = read_image_rgb(paths[0])
        h, w = first.shape[-2:]
        target_h, target_w = self.image_size
        scale = max(target_h / max(h, 1), target_w / max(w, 1), 1.0)
        h2, w2 = int(round(h * scale)), int(round(w * scale))
        if self.random_crop:
            top = random.randint(0, max(h2 - target_h, 0))
            left = random.randint(0, max(w2 - target_w, 0))
        else:
            top = max((h2 - target_h) // 2, 0)
            left = max((w2 - target_w) // 2, 0)
        crop = (top, left, h2, w2)
        return torch.stack([self._load_frame(path, crop) for path in paths], dim=0)


def parse_val_conditions(raw: str) -> list[tuple[float, float, float]]:
    conditions = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(v.strip()) for v in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"validation condition must be snr,bw,plr, got {item}")
        snr, bw, plr = parts
        if bw <= 0 or not 0 <= plr <= 1:
            raise ValueError(f"invalid validation condition: {item}")
        conditions.append((snr, bw, plr))
    if not conditions:
        raise ValueError("at least one validation condition is required")
    return conditions


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).detach().to(torch.float32).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred.flatten(0, 1), 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target.flatten(0, 1), 11, stride=1, padding=5)
    sig_x = F.avg_pool2d(pred.flatten(0, 1).square(), 11, 1, 5) - mu_x.square()
    sig_y = F.avg_pool2d(target.flatten(0, 1).square(), 11, 1, 5) - mu_y.square()
    sig_xy = F.avg_pool2d(pred.flatten(0, 1) * target.flatten(0, 1), 11, 1, 5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sig_x + sig_y + c2)
    )
    return float(ssim.mean().detach().cpu().item())


def mean_frame_metric(frame_outputs, name: str) -> float:
    vals = [out.metrics[name] for out in frame_outputs if name in out.metrics]
    return float(sum(vals) / len(vals)) if vals else 0.0


def sample_conditions(args: argparse.Namespace, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    snr = torch.empty(batch_size, device=device).uniform_(args.snr_min, args.snr_max)
    bw = torch.empty(batch_size, device=device).uniform_(args.bandwidth_min_mbps, args.bandwidth_max_mbps)
    plr = torch.empty(batch_size, device=device).uniform_(0.0, args.pkt_loss_max)
    return snr, bw, plr


def build_model(args: argparse.Namespace, device: torch.device) -> SatelliteDCVCFM:
    if not (args.model_path_i or args.model_path_p or args.resume):
        LOGGER.warning(
            "No pretrained DCVC-FM weights given (--model_path_i / --model_path_p) and no "
            "--resume checkpoint. The DCVC-FM backbone is RANDOMLY INITIALIZED, so Stage B/C "
            "training and any reported PSNR/bpp will be meaningless. Pass the official DCVC-FM "
            "I-frame and P-frame checkpoints before training for real."
        )
    model = SatelliteDCVCFM.from_pretrained(
        model_path_i=args.model_path_i or None,
        model_path_p=args.model_path_p or None,
        strict=not args.non_strict_load,
        channel_type=args.channel_type,
        slot_num=args.num_slots,
        slot_dim=args.slot_dim,
        slot_output_dim=args.slot_output_dim,
        slot_iterations=args.slot_iterations,
        slot_adapter_resolution=(args.slot_adapter_h, args.slot_adapter_w),
        update_slots_on_p=not args.no_update_slots_on_p,
        enable_slot_modulation=args.enable_slot_modulation,
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
            "q_index_mode": args.q_index_mode,
            "q_rd_table_path": args.q_rd_table_path,
            "q_delta_max": args.q_delta_max,
            "q_mlp_hidden": args.q_mlp_hidden,
            "lambda_rate_over": args.capacity_lambda_rate_over,
            "lambda_rate_under": args.capacity_lambda_rate_under,
            "learnable_offsets": not args.disable_learnable_capacity_offsets,
        },
        selector_kwargs={
            "w_mag": args.w_mag,
            "w_novelty": args.w_novelty,
            "w_obj": args.w_obj,
            "w_temporal": args.w_temporal,
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
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        state = payload.get("model_state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        LOGGER.info("resume loaded: missing=%d unexpected=%d", len(missing), len(unexpected))
    return model


def configure_stage(model: SatelliteDCVCFM, args: argparse.Namespace) -> list[torch.nn.Parameter]:
    if args.stage == "A":
        for param in model.parameters():
            param.requires_grad = False
    elif args.stage == "B":
        model.freeze_dcvcfm_backbone()
    elif args.stage == "C":
        model.unfreeze_stage_c()
    else:
        raise ValueError(f"unknown stage: {args.stage}")
    return [p for p in model.parameters() if p.requires_grad]


def compute_training_loss(
    model: SatelliteDCVCFM,
    clip: torch.Tensor,
    *,
    snr: torch.Tensor,
    bw: torch.Tensor,
    plr: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    out = model.forward_sequence(
        clip,
        snr_db=snr,
        bandwidth_mbps=bw,
        packet_loss_rate=plr,
        q_index_i=args.q_index_i,
        q_index_p=None if args.capacity_q_index else args.q_index_p,
        intra_period=args.intra_period,
        enable_satellite=args.stage != "A",
    )
    recon = out["x_hat"]
    recon_loss = reconstruction_distortion(
        recon,
        clip,
        mse_weight=args.lambda_mse,
        l1_weight=args.lambda_l1,
    )
    rate_losses = []
    token_losses = []
    q_losses = []
    for frame in out["frames"]:
        if frame.frame_type != "P" or frame.control is None:
            continue
        rate_loss, _ = rate_budget_interval_loss(frame.bpp, frame.control)
        rate_losses.append(rate_loss)
        token_losses.append(token_selection_regularization(frame.selection, frame.control))
        q_losses.append(q_index_proxy_loss(frame.control, delta_weight=args.q_delta_reg_weight))
    rate_budget = torch.stack(rate_losses).mean() if rate_losses else recon.new_tensor(0.0)
    token_reg = torch.stack(token_losses).mean() if token_losses else recon.new_tensor(0.0)
    q_reg = torch.stack(q_losses).mean() if q_losses else recon.new_tensor(0.0)
    temporal = recon.new_tensor(0.0)
    for idx in range(1, clip.shape[1]):
        temporal = temporal + temporal_consistency_loss(
            recon[:, idx - 1], recon[:, idx], clip[:, idx - 1], clip[:, idx]
        )
    temporal = temporal / max(clip.shape[1] - 1, 1)

    slot_losses = []
    seen_slots: set[int] = set()
    for frame in out["frames"]:
        slot = frame.state.slot if frame.state is not None else None
        if slot is None or id(slot) in seen_slots:
            continue
        seen_slots.add(id(slot))
        slot_loss = slot_reconstruction_loss(slot)
        if slot_loss is not None:
            slot_losses.append(slot_loss)
    slot_recon = torch.stack(slot_losses).mean() if slot_losses else recon.new_tensor(0.0)

    total = (
        args.lambda_recon * recon_loss
        + args.lambda_rate_budget * rate_budget
        + args.lambda_temporal * temporal
        + args.lambda_token * token_reg
        + args.lambda_q_index * q_reg
        + args.lambda_slot_recon * slot_recon
    )
    logs = {
        "loss": float(total.detach().cpu().item()),
        "recon": float(recon_loss.detach().cpu().item()),
        "weighted_recon": float((args.lambda_recon * recon_loss).detach().cpu().item()),
        "rate_budget": float(rate_budget.detach().cpu().item()),
        "weighted_rate": float((args.lambda_rate_budget * rate_budget).detach().cpu().item()),
        "temporal": float(temporal.detach().cpu().item()),
        "token": float(token_reg.detach().cpu().item()),
        "q_index_loss": float(q_reg.detach().cpu().item()),
        "weighted_q_index": float((args.lambda_q_index * q_reg).detach().cpu().item()),
        "slot_recon": float(slot_recon.detach().cpu().item()),
        "psnr": compute_psnr(recon, clip),
        "ssim": compute_ssim_torch(recon, clip),
        "bpp": mean_frame_metric(out["frames"], "bpp"),
        "target_bpp": mean_frame_metric(out["frames"], "target_bpp"),
        "capacity": mean_frame_metric(out["frames"], "capacity_mbps"),
        "q_index": mean_frame_metric(out["frames"], "q_index"),
        "q_index_base": mean_frame_metric(out["frames"], "q_index_base"),
        "q_index_delta": mean_frame_metric(out["frames"], "q_index_delta"),
        "q_proxy_bpp": mean_frame_metric(out["frames"], "q_proxy_bpp"),
        "keep": mean_frame_metric(out["frames"], "keep_ratio"),
        "base": mean_frame_metric(out["frames"], "base_layer_ratio"),
        "enh": mean_frame_metric(out["frames"], "enhancement_layer_ratio"),
        "tx_ms": mean_frame_metric(out["frames"], "tx_time_ms"),
        "proc_ms": mean_frame_metric(out["frames"], "proc_time_ms"),
    }
    return total, logs, out


@torch.no_grad()
def validate(
    model: SatelliteDCVCFM,
    loader: DataLoader,
    *,
    conditions: list[tuple[float, float, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    per_condition = []
    for snr, bw, plr in conditions:
        sums: dict[str, float] = {}
        count = 0
        for batch_idx, clip in enumerate(loader):
            if args.val_max_batches > 0 and batch_idx >= args.val_max_batches:
                break
            clip = clip.to(device, non_blocking=True)
            out = model.forward_sequence(
                clip,
                snr_db=snr,
                bandwidth_mbps=bw,
                packet_loss_rate=plr,
                q_index_i=args.q_index_i,
                q_index_p=None if args.capacity_q_index else args.q_index_p,
                intra_period=args.intra_period,
                enable_satellite=args.stage != "A",
            )
            recon = out["x_hat"]
            metrics = {
                "PSNR": compute_psnr(recon, clip),
                "SSIM": compute_ssim_torch(recon, clip),
                "bpp": mean_frame_metric(out["frames"], "bpp"),
                "target_bpp": mean_frame_metric(out["frames"], "target_bpp"),
                "capacity": mean_frame_metric(out["frames"], "capacity_mbps"),
                "q_index": mean_frame_metric(out["frames"], "q_index"),
                "q_index_base": mean_frame_metric(out["frames"], "q_index_base"),
                "q_index_delta": mean_frame_metric(out["frames"], "q_index_delta"),
                "keep_ratio": mean_frame_metric(out["frames"], "keep_ratio"),
                "base_layer_ratio": mean_frame_metric(out["frames"], "base_layer_ratio"),
                "enhancement_layer_ratio": mean_frame_metric(out["frames"], "enhancement_layer_ratio"),
            }
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
        averaged = {key: value / max(count, 1) for key, value in sums.items()}
        averaged.update({"snr_db": snr, "bandwidth_mbps": bw, "packet_loss_rate": plr})
        per_condition.append(averaged)

    mean_psnr = sum(item.get("PSNR", 0.0) for item in per_condition) / max(len(per_condition), 1)
    mean_bpp = sum(item.get("bpp", 0.0) for item in per_condition) / max(len(per_condition), 1)
    score = -mean_psnr + args.best_bpp_weight * mean_bpp
    summary = {
        "score": score,
        "mean_PSNR": mean_psnr,
        "mean_bpp": mean_bpp,
        "conditions": per_condition,
    }
    return score, summary


def save_checkpoint(
    path: Path,
    *,
    model: SatelliteDCVCFM,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    best_score: float,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
        "best_score": best_score,
        "extra": extra or {},
    }
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_root = Path(args.data_dir) / "train" if (Path(args.data_dir) / "train").exists() else Path(args.data_dir)
    data_root_for_val = Path(args.data_dir)
    if args.val_dir:
        val_root = Path(args.val_dir)
    elif (data_root_for_val / "val").exists():
        val_root = data_root_for_val / "val"
    elif (data_root_for_val / "test").exists():
        val_root = data_root_for_val / "test"
    else:
        val_root = data_root_for_val / "val"
    train_ds = VideoFolderDataset(
        train_root,
        clip_len=args.clip_len,
        image_size=(args.img_h, args.img_w),
        stride=args.clip_stride,
        max_clips=args.max_train_clips,
        random_crop=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if val_root.exists():
        val_ds = VideoFolderDataset(
            val_root,
            clip_len=args.clip_len,
            image_size=(args.img_h, args.img_w),
            stride=args.clip_len - 1,
            max_clips=args.max_val_clips,
            random_crop=False,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args, device)
    params = configure_stage(model, args)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay) if params else None
    conditions = parse_val_conditions(args.val_conditions)

    best_score = float("inf")
    global_step = 0
    if val_loader is not None:
        init_score, init_summary = validate(model, val_loader, conditions=conditions, args=args, device=device)
    else:
        init_score, init_summary = 0.0, {"note": "no validation loader available"}
    best_score = init_score
    save_checkpoint(
        save_dir / "best.pt",
        model=model,
        args=args,
        step=0,
        epoch=0,
        best_score=best_score,
        extra={"initial_validation": init_summary},
    )
    LOGGER.info("saved initial best baseline: %s | score=%.4f | %s", save_dir / "best.pt", best_score, init_summary)

    if args.stage == "A" or args.max_steps == 0:
        save_checkpoint(save_dir / "final.pt", model=model, args=args, step=0, epoch=0, best_score=best_score)
        LOGGER.info("Stage A / eval-only run complete.")
        return

    amp_enabled = device.type == "cuda" and args.amp and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for clip in train_loader:
            global_step += 1
            clip = clip.to(device, non_blocking=True)
            snr, bw, plr = sample_conditions(args, clip.shape[0], device)
            start = time.perf_counter()
            if optimizer is None:
                raise RuntimeError("no trainable parameters for this stage")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss, logs, _ = compute_training_loss(model, clip, snr=snr, bw=bw, plr=plr, args=args)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if global_step % args.log_interval == 0:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                LOGGER.info(
                    "step=%d loss=%.4f recon=%.4f weighted_rate=%.4f slot=%.4f "
                    "PSNR=%.2f SSIM=%.4f actual_bpp=%.4f target_bpp=%.4f "
                    "capacity=%.2f q=%.1f q_base=%.1f q_delta=%.2f keep=%.3f "
                    "base=%.3f enh=%.3f tx=%.2fms proc=%.2fms step=%.1fms",
                    global_step,
                    logs["loss"],
                    logs["weighted_recon"],
                    logs["weighted_rate"],
                    logs["slot_recon"],
                    logs["psnr"],
                    logs["ssim"],
                    logs["bpp"],
                    logs["target_bpp"],
                    logs["capacity"],
                    logs["q_index"],
                    logs["q_index_base"],
                    logs["q_index_delta"],
                    logs["keep"],
                    logs["base"],
                    logs["enh"],
                    logs["tx_ms"],
                    logs["proc_ms"],
                    elapsed_ms,
                )

            if val_loader is not None and args.val_interval > 0 and global_step % args.val_interval == 0:
                score, summary = validate(model, val_loader, conditions=conditions, args=args, device=device)
                for item in summary["conditions"]:
                    LOGGER.info(
                        "val SNR=%.1f BW=%.1f PLR=%.2f PSNR=%.2f SSIM=%.4f bpp=%.4f "
                        "target=%.4f q=%.1f keep=%.3f",
                        item["snr_db"],
                        item["bandwidth_mbps"],
                        item["packet_loss_rate"],
                        item.get("PSNR", 0.0),
                        item.get("SSIM", 0.0),
                        item.get("bpp", 0.0),
                        item.get("target_bpp", 0.0),
                        item.get("q_index", 0.0),
                        item.get("keep_ratio", 0.0),
                    )
                if score < best_score:
                    best_score = score
                    save_checkpoint(
                        save_dir / "best.pt",
                        model=model,
                        args=args,
                        step=global_step,
                        epoch=epoch,
                        best_score=best_score,
                        extra={"validation": summary},
                    )
                    LOGGER.info("new best saved: score=%.4f", best_score)
                model.train()

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_checkpoint(
                    save_dir / f"step_{global_step:07d}.pt",
                    model=model,
                    args=args,
                    step=global_step,
                    epoch=epoch,
                    best_score=best_score,
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint(save_dir / "final.pt", model=model, args=args, step=global_step, epoch=epoch, best_score=best_score)
    LOGGER.info("training complete: final=%s best=%s best_score=%.4f", save_dir / "final.pt", save_dir / "best.pt", best_score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train satellite-aware adapters for DCVC-FM")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, default="")
    p.add_argument("--model_path_i", type=str, default="")
    p.add_argument("--model_path_p", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_dir", type=str, default="checkpoints/dcvcfm_satellite")
    p.add_argument("--stage", type=str, default="B", choices=["A", "B", "C"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--non_strict_load", action="store_true")
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--clip_len", type=int, default=5)
    p.add_argument("--clip_stride", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train_clips", type=int, default=0)
    p.add_argument("--max_val_clips", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="enable CUDA AMP; default is fp32 for entropy stability")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--val_max_batches", type=int, default=16)
    p.add_argument("--val_conditions", type=str, default=DEFAULT_VAL_CONDITIONS)
    p.add_argument("--best_bpp_weight", type=float, default=8.0)
    p.add_argument("--channel_type", type=str, default="satellite", choices=["identity", "awgn", "rayleigh", "satellite"])
    p.add_argument("--snr_min", type=float, default=1.0)
    p.add_argument("--snr_max", type=float, default=25.0)
    p.add_argument("--bandwidth_min_mbps", type=float, default=1.0)
    p.add_argument("--bandwidth_max_mbps", type=float, default=25.0)
    p.add_argument("--pkt_loss_max", type=float, default=0.5)
    p.add_argument("--num_slots", type=int, default=7)
    p.add_argument("--slot_dim", type=int, default=64)
    p.add_argument("--slot_output_dim", type=int, default=128)
    p.add_argument("--slot_iterations", type=int, default=3)
    p.add_argument("--slot_adapter_h", type=int, default=128)
    p.add_argument("--slot_adapter_w", type=int, default=128)
    p.add_argument("--no_update_slots_on_p", action="store_true")
    p.set_defaults(enable_slot_modulation=True)
    p.add_argument("--enable_slot_modulation", dest="enable_slot_modulation", action="store_true",
                   help="Slot->decoder FiLM modulation (on by default, matches the curriculum trainer)")
    p.add_argument("--disable_slot_modulation", dest="enable_slot_modulation", action="store_false",
                   help="ablation: turn off Slot->decoder FiLM modulation")
    p.add_argument("--w_mag", type=float, default=0.45)
    p.add_argument("--w_novelty", type=float, default=0.25)
    p.add_argument("--w_obj", type=float, default=0.20)
    p.add_argument("--w_temporal", type=float, default=0.10)
    p.add_argument("--ste_temperature", type=float, default=0.08)
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
    p.add_argument("--q_index_mode", type=str, default="rd_table", choices=["linear", "rd_table"],
                   help="rd_table requires --q_rd_table_path (no silent fallback to linear)")
    p.add_argument("--q_rd_table_path", type=str, default="")
    p.add_argument("--q_delta_max", type=float, default=0.0)
    p.add_argument("--q_mlp_hidden", type=int, default=32)
    p.add_argument("--capacity_lambda_rate_over", type=float, default=1.0)
    p.add_argument("--capacity_lambda_rate_under", type=float, default=0.35)
    p.set_defaults(disable_learnable_capacity_offsets=True)
    p.add_argument("--disable_learnable_capacity_offsets", action="store_true")
    p.add_argument("--learnable_capacity_offsets", dest="disable_learnable_capacity_offsets", action="store_false")
    p.add_argument("--no_row_packet_loss", action="store_true")
    p.add_argument("--base_snr_gain_db", type=float, default=6.0)
    p.add_argument("--enhancement_snr_gain_db", type=float, default=0.0)
    p.add_argument("--base_plr_scale", type=float, default=0.25)
    p.add_argument("--enhancement_plr_scale", type=float, default=1.25)
    p.add_argument("--rician_k_db", type=float, default=10.0)
    p.add_argument("--shadowing_std_db", type=float, default=3.0)
    p.add_argument("--beam_switch_prob", type=float, default=0.01)
    p.set_defaults(capacity_q_index=True)
    p.add_argument("--fixed_q_index", dest="capacity_q_index", action="store_false",
                   help="use --q_index_p instead of capacity-controlled q index")
    p.add_argument("--q_index_i", type=int, default=63)
    p.add_argument("--q_index_p", type=int, default=63)
    p.add_argument("--intra_period", type=int, default=9999)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=0.05)
    # Rate is owned by the deterministic q_index path; the rate-budget interval
    # loss is OFF by default (kept tunable for ablation only).
    p.add_argument("--lambda_rate_budget", type=float, default=0.0)
    p.add_argument("--lambda_temporal", type=float, default=0.05)
    p.add_argument("--lambda_token", type=float, default=0.05)
    p.add_argument("--lambda_q_index", type=float, default=0.0)
    p.add_argument("--q_delta_reg_weight", type=float, default=0.02)
    p.add_argument("--lambda_slot_recon", type=float, default=0.1,
                   help="weight for the Slot Attention object-discovery reconstruction loss")
    return p.parse_args()


if __name__ == "__main__":
    main()
