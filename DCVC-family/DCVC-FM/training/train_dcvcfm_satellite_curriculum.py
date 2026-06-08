"""Curriculum training entry point for satellite-aware DCVC-FM.

This script is the recommended training path for the satellite extension.  The
older `train_dcvcfm_satellite.py` remains available as a compact A/B/C runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.satellite_curriculum import (  # noqa: E402
    PHASES,
    compute_satellite_sequence_loss,
    configure_trainable_parameters,
    get_phase_spec,
    parse_float_list,
    repeat_clip_for_paired_bandwidth,
    sample_conditions_for_phase,
    slot_adapter_auxiliary_loss,
)
from training.train_dcvcfm_satellite import (  # noqa: E402
    DEFAULT_VAL_CONDITIONS,
    VideoFolderDataset,
    build_model,
    compute_psnr,
    compute_ssim_torch,
    parse_val_conditions,
    save_checkpoint,
    validate,
)


LOGGER = logging.getLogger("train_dcvcfm_satellite_curriculum")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def init_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    """Initialise torch.distributed when launched via torchrun.

    Returns ``(distributed, rank, world_size, local_rank)``.  When the process
    is not launched with ``WORLD_SIZE>1`` this is a no-op and the single-GPU
    code path is preserved byte-for-byte.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if world_size <= 1:
        return False, 0, 1, 0
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def broadcast_module(model: torch.nn.Module) -> None:
    """Broadcast all parameters/buffers from rank 0 so every replica starts identical."""
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=0)


def average_gradients(params: list[torch.nn.Parameter], world_size: int) -> None:
    """Average gradients across replicas (manual DDP suited to the custom forward loop).

    Every rank must launch the all-reduce for the *same* parameters in the *same*
    order, otherwise a parameter that is used on one rank but not another (the
    selection masks are data-dependent) would dead-lock the collective.  We
    therefore materialise a zero gradient for any param that did not receive one.
    """
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


def set_phase_defaults(args: argparse.Namespace) -> None:
    spec = get_phase_spec(args.phase)
    args.stage = spec.stage_alias
    if args.channel_type == "auto":
        args.channel_type = spec.channel_type_hint
    if args.phase == "slot_warmup":
        # Slot warm-up is pure object discovery: no rate/channel/selection terms.
        args.lambda_rate_budget = 0.0
        args.lambda_token = 0.0
        args.lambda_monotonic = 0.0
        args.lambda_channel = 0.0
    elif args.phase == "capacity_calibration":
        # Deprecated phase kept for ablation only; channel teacher stays off here.
        args.lambda_channel = 0.0
    # robust_curriculum / joint_finetune: respect the exact --lambda_channel passed
    # on the command line (no hidden floor), so the channel-teacher weight can be
    # tuned down to reduce coupling / training difficulty as intended.


def update_learning_rate(optimizer: torch.optim.Optimizer, args: argparse.Namespace, step: int) -> float:
    """Apply the Slot-Attention-style warmup and exponential decay schedule."""

    lr = float(args.lr)
    if args.lr_warmup_steps > 0:
        lr *= min(1.0, float(step) / float(args.lr_warmup_steps))
    if args.lr_decay_steps > 0 and args.lr_decay_rate > 0:
        lr *= float(args.lr_decay_rate) ** (float(step) / float(args.lr_decay_steps))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    data_root = Path(args.data_dir)
    train_root = data_root / "train" if (data_root / "train").exists() else data_root
    if args.val_dir:
        val_root = Path(args.val_dir)
    elif (data_root / "val").exists():
        val_root = data_root / "val"
    elif (data_root / "test").exists():
        val_root = data_root / "test"
    else:
        val_root = data_root / "val"

    train_ds = VideoFolderDataset(
        train_root,
        clip_len=args.clip_len,
        image_size=(args.img_h, args.img_w),
        stride=args.clip_stride,
        max_clips=args.max_train_clips,
        random_crop=True,
    )
    train_sampler: DistributedSampler | None = None
    if distributed:
        # drop_last keeps every replica at an identical batch count, which keeps
        # the manual all-reduce / validation barriers perfectly in lock-step.
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
            stride=max(1, args.clip_len - 1),
            max_clips=args.max_val_clips,
            random_crop=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    return train_loader, val_loader, train_sampler


def slot_warmup_loss(model, clip: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    b, t, c, h, w = clip.shape
    frames = clip.reshape(b * t, c, h, w)
    if args.slot_frames_per_clip > 0 and frames.shape[0] > args.slot_frames_per_clip:
        idx = torch.randperm(frames.shape[0], device=frames.device)[: args.slot_frames_per_clip]
        frames = frames[idx]

    slot_out = model.slot_adapter(frames)
    slot_loss, logs = slot_adapter_auxiliary_loss(
        slot_out,
        frames,
        recon_weight=args.lambda_slot_recon,
        entropy_weight=args.lambda_slot_entropy,
        balance_weight=args.lambda_slot_balance,
    )

    temporal = frames.new_tensor(0.0)
    if args.lambda_slot_temporal > 0 and args.slot_frames_per_clip <= 0:
        object_map = slot_out.object_importance.reshape(b, t, 1, h, w)
        if t > 1:
            temporal = F.l1_loss(object_map[:, 1:], object_map[:, :-1].detach())

    total = args.lambda_slot * slot_loss + args.lambda_slot_temporal * temporal
    recon = slot_out.recon_image.clamp(0.0, 1.0)
    logs.update(
        {
            "loss": float(total.detach().cpu().item()),
            "slot_temporal": float(temporal.detach().cpu().item()),
            "psnr": compute_psnr(recon, frames),
            "ssim": compute_ssim_torch(recon.unsqueeze(1), frames.unsqueeze(1)),
            "bpp": 0.0,
            "target_bpp": 0.0,
            "capacity": 0.0,
            "keep": 0.0,
            "base": 0.0,
            "enh": 0.0,
            "tx_ms": 0.0,
            "proc_ms": 0.0,
        }
    )
    return total, logs, {"slot": slot_out}


@torch.no_grad()
def validate_slot_warmup(model, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> tuple[float, dict[str, Any]]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, clip in enumerate(loader):
        if args.val_max_batches > 0 and batch_idx >= args.val_max_batches:
            break
        clip = clip.to(device, non_blocking=True)
        loss, logs, _ = slot_warmup_loss(model, clip, args)
        for key, value in logs.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        sums["raw_loss"] = sums.get("raw_loss", 0.0) + float(loss.detach().cpu().item())
        count += 1
    averaged = {key: value / max(count, 1) for key, value in sums.items()}
    score = averaged.get("raw_loss", averaged.get("loss", 0.0))
    return score, {"score": score, "slot_validation": averaged}


@torch.no_grad()
def validate_for_phase(
    model,
    val_loader: DataLoader | None,
    *,
    args: argparse.Namespace,
    conditions: list[tuple[float, float, float]],
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    if val_loader is None:
        return 0.0, {"note": "no validation loader available"}
    if args.phase == "slot_warmup":
        return validate_slot_warmup(model, val_loader, args, device)
    return validate(model, val_loader, conditions=conditions, args=args, device=device)


def prepare_clip_and_conditions(
    clip: torch.Tensor,
    args: argparse.Namespace,
    *,
    device: torch.device,
    global_step: int,
) -> tuple[torch.Tensor, Any]:
    spec = get_phase_spec(args.phase)
    clip = clip.to(device, non_blocking=True)
    if spec.paired_bandwidth:
        clip = repeat_clip_for_paired_bandwidth(clip, args.bandwidth_grid)
    conditions = sample_conditions_for_phase(
        args,
        phase=args.phase,
        batch_size=clip.shape[0],
        device=device,
        global_step=global_step,
        max_steps=args.max_steps,
    )
    return clip, conditions


def log_validation_summary(summary: dict[str, Any]) -> None:
    for item in summary.get("conditions", []):
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
    if "slot_validation" in summary:
        item = summary["slot_validation"]
        LOGGER.info(
            "slot-val loss=%.5f PSNR=%.2f SSIM=%.4f recon=%.5f entropy=%.5f balance=%.5f",
            item.get("loss", 0.0),
            item.get("psnr", 0.0),
            item.get("ssim", 0.0),
            item.get("slot_recon", 0.0),
            item.get("slot_entropy", 0.0),
            item.get("slot_balance", 0.0),
        )


def train(args: argparse.Namespace) -> None:
    set_phase_defaults(args)
    distributed, rank, world_size, local_rank = init_distributed(args)
    is_main = rank == 0
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{rank}] %(message)s",
    )
    if distributed and torch.cuda.is_available() and args.device != "cpu":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    # Identical seed across replicas -> identical model init; the DistributedSampler
    # handles per-rank data sharding instead of seed offsets.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if distributed:
        LOGGER.info("distributed run: world_size=%d rank=%d local_rank=%d device=%s", world_size, rank, local_rank, device)
        if args.amp and not args.no_amp:
            raise RuntimeError("multi-GPU curriculum training keeps AMP disabled; remove --amp for torchrun runs")

    spec = get_phase_spec(args.phase)
    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (save_dir / "phase.json").write_text(json.dumps(spec.__dict__, indent=2), encoding="utf-8")
    if distributed:
        dist.barrier()

    train_loader, val_loader, train_sampler = build_loaders(
        args, device, distributed=distributed, rank=rank, world_size=world_size
    )
    model = build_model(args, device)
    if distributed:
        broadcast_module(model)
    params = configure_trainable_parameters(model, args.phase)
    LOGGER.info("phase=%s | %s", spec.name, spec.description)
    LOGGER.info("channel_type=%s trainable_params=%d", args.channel_type, sum(p.numel() for p in params))

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay) if params else None
    conditions = parse_val_conditions(args.val_conditions)

    if is_main:
        best_score, init_summary = validate_for_phase(model, val_loader, args=args, conditions=conditions, device=device)
        save_checkpoint(
            save_dir / "best.pt",
            model=model,
            args=args,
            step=0,
            epoch=0,
            best_score=best_score,
            extra={"phase": spec.__dict__, "initial_validation": init_summary},
        )
        LOGGER.info("saved initial best baseline: %s | score=%.5f", save_dir / "best.pt", best_score)
        log_validation_summary(init_summary)
    else:
        best_score = float("inf")
    if distributed:
        score_t = torch.tensor([best_score], dtype=torch.float64, device=device)
        dist.broadcast(score_t, src=0)
        best_score = float(score_t.item())
        dist.barrier()

    if args.phase == "baseline" or args.max_steps == 0:
        if is_main:
            save_checkpoint(
                save_dir / "final.pt",
                model=model,
                args=args,
                step=0,
                epoch=0,
                best_score=best_score,
                extra={"phase": spec.__dict__},
            )
            LOGGER.info("baseline/eval-only run complete.")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    if optimizer is None:
        raise RuntimeError(f"phase {args.phase} has no trainable parameters")

    amp_enabled = device.type == "cuda" and args.amp and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    global_step = 0
    last_epoch = 0
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for clip in train_loader:
            global_step += 1
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            current_lr = update_learning_rate(optimizer, args, global_step)

            if args.phase == "slot_warmup":
                clip = clip.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss, logs, _ = slot_warmup_loss(model, clip, args)
            else:
                clip, cond = prepare_clip_and_conditions(clip, args, device=device, global_step=global_step)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss, logs, out = compute_satellite_sequence_loss(
                        model,
                        clip,
                        conditions=cond,
                        args=args,
                        enable_satellite=spec.enable_satellite,
                        include_channel_teacher=args.phase in {"robust_curriculum", "joint_finetune"},
                    )
                recon = out["x_hat"].detach().clamp(0, 1)
                logs["psnr"] = compute_psnr(recon, clip)
                logs["ssim"] = compute_ssim_torch(recon, clip)

            if not loss.requires_grad:
                raise RuntimeError(
                    "training loss is not connected to trainable parameters; "
                    "check phase trainability and differentiable masks"
                )
            scaler.scale(loss).backward()
            if distributed:
                # Average gradients of the (scaled) loss across replicas; the scale
                # is identical on every rank so unscaling afterwards stays correct.
                average_gradients(params, world_size)
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if is_main and global_step % args.log_interval == 0:
                elapsed_ms = (time.perf_counter() - begin) * 1000.0
                LOGGER.info(
                    "phase=%s step=%d loss=%.5f recon=%.5f rate=%.5f mono=%.5f ch=%.5f "
                    "PSNR=%.2f SSIM=%.4f actual_bpp=%.4f target_bpp=%.4f capacity=%.2f "
                    "q=%.1f q_base=%.1f q_delta=%.2f keep=%.3f base=%.3f enh=%.3f "
                    "tx=%.2fms proc=%.2fms step=%.1fms",
                    args.phase,
                    global_step,
                    logs.get("loss", 0.0),
                    logs.get("weighted_recon", logs.get("slot_recon", 0.0)),
                    logs.get("weighted_rate", 0.0),
                    logs.get("monotonic", 0.0),
                    logs.get("channel", 0.0),
                    logs.get("psnr", 0.0),
                    logs.get("ssim", 0.0),
                    logs.get("bpp", 0.0),
                    logs.get("target_bpp", 0.0),
                    logs.get("capacity", 0.0),
                    logs.get("q_index", 0.0),
                    logs.get("q_index_base", 0.0),
                    logs.get("q_index_delta", 0.0),
                    logs.get("keep", 0.0),
                    logs.get("base", 0.0),
                    logs.get("enh", 0.0),
                    logs.get("tx_ms", 0.0),
                    logs.get("proc_ms", 0.0),
                    elapsed_ms,
                )
                LOGGER.info("lr=%.8f", current_lr)

            if val_loader is not None and args.val_interval > 0 and global_step % args.val_interval == 0:
                # All ranks reach the same step (DistributedSampler keeps batch counts
                # equal); only rank 0 evaluates while the others wait on the barriers.
                if distributed:
                    dist.barrier()
                if is_main:
                    score, summary = validate_for_phase(model, val_loader, args=args, conditions=conditions, device=device)
                    log_validation_summary(summary)
                    if score < best_score:
                        best_score = score
                        save_checkpoint(
                            save_dir / "best.pt",
                            model=model,
                            args=args,
                            step=global_step,
                            epoch=epoch,
                            best_score=best_score,
                            extra={"phase": spec.__dict__, "validation": summary},
                        )
                        LOGGER.info("new best saved: score=%.5f", best_score)
                model.train()
                if distributed:
                    dist.barrier()

            if is_main and args.save_interval > 0 and global_step % args.save_interval == 0:
                save_checkpoint(
                    save_dir / f"step_{global_step:07d}.pt",
                    model=model,
                    args=args,
                    step=global_step,
                    epoch=epoch,
                    best_score=best_score,
                    extra={"phase": spec.__dict__},
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if is_main:
        save_checkpoint(
            save_dir / "final.pt",
            model=model,
            args=args,
            step=global_step,
            epoch=last_epoch,
            best_score=best_score,
            extra={"phase": spec.__dict__},
        )
        LOGGER.info("training complete: final=%s best=%s best_score=%.5f", save_dir / "final.pt", save_dir / "best.pt", best_score)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curriculum train satellite-aware DCVC-FM")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, default="")
    p.add_argument("--model_path_i", type=str, default="")
    p.add_argument("--model_path_p", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_dir", type=str, default="checkpoints/dcvcfm_satellite_curriculum")
    p.add_argument("--phase", type=str, default="selection_warmup", choices=PHASES)
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
    p.add_argument("--lr_warmup_steps", type=int, default=0,
                   help="linear warmup steps; follows the official Slot Attention schedule when >0")
    p.add_argument("--lr_decay_steps", type=int, default=0,
                   help="exponential decay denominator; disabled when <=0")
    p.add_argument("--lr_decay_rate", type=float, default=1.0,
                   help="exponential decay rate, e.g. 0.5 as in official Slot Attention")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="enable CUDA AMP; fp32 is default for entropy stability")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--val_max_batches", type=int, default=16)
    p.add_argument("--val_conditions", type=str, default=DEFAULT_VAL_CONDITIONS)
    p.add_argument("--best_bpp_weight", type=float, default=8.0)

    p.add_argument("--channel_type", type=str, default="auto", choices=["auto", "identity", "awgn", "rayleigh", "satellite"])
    p.add_argument("--snr_min", type=float, default=1.0)
    p.add_argument("--snr_max", type=float, default=25.0)
    p.add_argument("--bandwidth_min_mbps", type=float, default=1.0)
    p.add_argument("--bandwidth_max_mbps", type=float, default=25.0)
    p.add_argument("--pkt_loss_max", type=float, default=0.5)
    p.add_argument("--bandwidth_grid", type=str, default="1,2,5,10,20,25")
    p.add_argument("--capacity_calibration_snr_db", type=float, default=20.0)
    p.add_argument("--capacity_calibration_plr", type=float, default=0.0)
    p.add_argument("--robust_snr_mid", type=float, default=10.0)
    p.add_argument("--robust_plr_warmup", type=float, default=0.05)

    p.add_argument("--num_slots", type=int, default=7)
    p.add_argument("--slot_dim", type=int, default=64)
    p.add_argument("--slot_output_dim", type=int, default=128)
    p.add_argument("--slot_iterations", type=int, default=3)
    p.add_argument("--slot_adapter_h", type=int, default=128)
    p.add_argument("--slot_adapter_w", type=int, default=128)
    p.add_argument("--slot_frames_per_clip", type=int, default=0)
    p.add_argument("--no_update_slots_on_p", action="store_true")
    p.set_defaults(enable_slot_modulation=True)
    p.add_argument("--enable_slot_modulation", dest="enable_slot_modulation", action="store_true",
                   help="Slot FiLM modulation of the decoded latent (key innovation); on by default")
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
                   help="primary rate knob: capacity->q_index via offline RD table (falls back to linear if no table)")
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
    p.add_argument("--fixed_q_index", dest="capacity_q_index", action="store_false")
    p.add_argument("--q_index_i", type=int, default=63)
    p.add_argument("--q_index_p", type=int, default=63)
    p.add_argument("--intra_period", type=int, default=9999)

    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=0.05)
    # Rate is owned by the deterministic q_index path, so the rate-budget interval
    # loss is OFF by default (set >0 only as a soft safety guard if ever needed).
    p.add_argument("--lambda_rate_budget", type=float, default=0.0)
    p.add_argument("--lambda_temporal", type=float, default=0.05)
    p.add_argument("--lambda_token", type=float, default=0.05)
    p.add_argument("--lambda_q_index", type=float, default=0.0)
    p.add_argument("--q_delta_reg_weight", type=float, default=0.02)
    p.add_argument("--lambda_monotonic", type=float, default=0.0)
    p.add_argument("--lambda_channel", type=float, default=0.0)
    p.add_argument("--lambda_slot", type=float, default=1.0)
    p.add_argument("--lambda_slot_recon", type=float, default=1.0)
    p.add_argument("--lambda_slot_entropy", type=float, default=0.03)
    p.add_argument("--lambda_slot_balance", type=float, default=0.05)
    p.add_argument("--lambda_slot_temporal", type=float, default=0.01)

    args = p.parse_args()
    parse_float_list(args.bandwidth_grid)
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
