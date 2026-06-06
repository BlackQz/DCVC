"""Prepare BVI-DVC 512 data for DCVC-FM satellite experiments.

The script accepts either 8-bit YUV420 files or existing PNG frame folders.  It
creates a standard frame dataset for satellite training/evaluation and generates
DCVC-FM `test_video.py` config files for the official baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.video_reader import YUVReader  # noqa: E402
from src.utils.video_writer import PNGWriter  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def safe_name(path: Path, used: set[str]) -> str:
    stem = path.stem if path.is_file() else path.name
    name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem)
    name = name.strip("._") or "seq"
    original = name
    idx = 1
    while name in used:
        idx += 1
        name = f"{original}_{idx:02d}"
    used.add(name)
    return name


def yuv_frame_count(path: Path, width: int, height: int, bit_depth: int) -> int:
    bytes_per_sample = 2 if bit_depth > 8 else 1
    frame_bytes = width * height * 3 // 2 * bytes_per_sample
    size = path.stat().st_size
    if frame_bytes <= 0 or size < frame_bytes:
        return 0
    return size // frame_bytes


def discover_yuv(root: Path, width: int, height: int, bit_depth: int) -> list[dict]:
    items = []
    used: set[str] = set()
    for path in sorted(root.rglob("*.yuv")):
        frames = yuv_frame_count(path, width, height, bit_depth)
        if frames <= 0:
            continue
        name = safe_name(path, used)
        items.append({"name": name, "path": path, "frames": frames, "width": width, "height": height})
    return items


def is_frame_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTS for child in path.iterdir())


def discover_png_dirs(root: Path, width: int, height: int) -> list[dict]:
    items = []
    used: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not is_frame_dir(path):
            continue
        images = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not images:
            continue
        with Image.open(images[0]) as img:
            w, h = img.size
        if width > 0 and height > 0 and (w != width or h != height):
            continue
        name = safe_name(path, used)
        items.append({"name": name, "path": path, "frames": len(images), "width": w, "height": h, "images": images})
    return items


def ensure_empty_or_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    else:
        raise ValueError(f"unknown split mode: {mode}")


def convert_yuv_to_png(seq: dict, dst_dir: Path, bit_depth: int, max_frames: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dst_dir.glob("im*.png"))
    if existing and (max_frames <= 0 or len(existing) >= min(seq["frames"], max_frames)):
        return len(existing)
    for old in existing:
        old.unlink()
    reader = YUVReader(str(seq["path"]), seq["width"], seq["height"], src_format="420", bit_depth=bit_depth)
    writer = PNGWriter(str(dst_dir), seq["width"], seq["height"])
    count = 0
    rgb = reader.read_one_frame(dst_format="rgb")
    while not reader.eof:
        writer.write_one_frame(rgb=rgb, src_format="rgb")
        count += 1
        if max_frames > 0 and count >= max_frames:
            break
        rgb = reader.read_one_frame(dst_format="rgb")
    reader.close()
    writer.close()
    return count


def standardize_png_dir(seq: dict, dst_dir: Path, mode: str, max_frames: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = list(seq["images"])
    if max_frames > 0:
        images = images[:max_frames]
    for idx, src in enumerate(images, start=1):
        dst = dst_dir / f"im{idx:05d}.png"
        link_or_copy(src, dst, mode)
    return len(images)


def split_sequences(seqs: list[dict], val_ratio: float, val_names: set[str]) -> tuple[list[dict], list[dict]]:
    if val_names:
        val = [s for s in seqs if s["name"] in val_names]
        train = [s for s in seqs if s["name"] not in val_names]
    else:
        val_count = max(1, int(round(len(seqs) * val_ratio))) if len(seqs) > 1 else 1
        val = seqs[-val_count:]
        train = seqs[:-val_count] if len(seqs) > 1 else seqs
    if not train:
        train = val
    return train, val


def create_split_frame_dirs(all_dir: Path, split_dir: Path, seqs: Iterable[dict], mode: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        src_dir = all_dir / seq["name"]
        dst_dir = split_dir / seq["name"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for frame in sorted(src_dir.glob("im*.png")):
            link_or_copy(frame, dst_dir / frame.name, mode)


def create_rgb_config(
    path: Path,
    frame_root: Path,
    val_seqs: list[dict],
    eval_frames: int,
    *,
    base_path: str = "val",
) -> None:
    sequences = {}
    for seq in val_seqs:
        frames = seq["frames"] if eval_frames <= 0 else min(seq["frames"], eval_frames)
        sequences[seq["name"]] = {
            "width": int(seq["width"]),
            "height": int(seq["height"]),
            "frames": int(frames),
            "intra_period": int(frames),
        }
    cfg = {
        "root_path": str(frame_root.resolve()),
        "test_classes": {
            "BVI-DVC-512": {
                "test": 1,
                "base_path": base_path,
                "src_type": "png",
                "sequences": sequences,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")


def create_existing_split_view(args: argparse.Namespace, input_root: Path, work_root: Path) -> None:
    train_root = input_root / args.train_dir_name
    val_dir_name = args.val_dir_name or args.test_dir_name
    val_root = input_root / val_dir_name
    if not train_root.exists():
        raise FileNotFoundError(f"train split not found: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"val split not found: {val_root}")

    train_seqs = discover_png_dirs(train_root, args.width, args.height)
    val_seqs = discover_png_dirs(val_root, args.width, args.height)
    if not train_seqs:
        raise FileNotFoundError(f"no PNG sequence folders found under {train_root}")
    if not val_seqs:
        raise FileNotFoundError(f"no PNG sequence folders found under {val_root}")

    config_dir = work_root / "configs"
    official_root = work_root / "official_rgb"
    official_val = official_root / val_dir_name
    if args.overwrite and official_val.exists():
        shutil.rmtree(official_val)
    official_val.mkdir(parents=True, exist_ok=True)

    standardized = []
    for seq in val_seqs:
        dst_dir = official_val / seq["name"]
        frame_count = standardize_png_dir(seq, dst_dir, args.split_mode, args.max_frames)
        standardized.append({
            "name": seq["name"],
            "width": seq["width"],
            "height": seq["height"],
            "frames": frame_count,
        })

    create_rgb_config(
        config_dir / "bvidvc512_rgb.json",
        official_root,
        standardized,
        args.eval_frames,
        base_path=val_dir_name,
    )
    manifest = {
        "input_root": str(input_root),
        "work_root": str(work_root),
        "source_type": "png_existing_split",
        "width": args.width,
        "height": args.height,
        "train_dir": str(train_root),
        "val_dir": str(val_root),
        "train_sequences": [s["name"] for s in train_seqs],
        "val_sequences": [s["name"] for s in standardized],
        "train_video_count": len(train_seqs),
        "val_video_count": len(standardized),
        "official_rgb_root": str(official_root),
        "rgb_config": str((config_dir / "bvidvc512_rgb.json").resolve()),
    }
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "manifest.json").write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    print(json.dumps(manifest, indent=4, ensure_ascii=False))


def create_yuv_config(path: Path, work_root: Path, yuv_dir: Path, val_seqs: list[dict], eval_frames: int) -> None:
    sequences = {}
    for seq in val_seqs:
        frames = seq["frames"] if eval_frames <= 0 else min(seq["frames"], eval_frames)
        sequences[f"{seq['name']}.yuv"] = {
            "width": int(seq["width"]),
            "height": int(seq["height"]),
            "frames": int(frames),
            "intra_period": -1,
        }
    cfg = {
        "root_path": str(work_root.resolve()),
        "test_classes": {
            "BVI-DVC-512": {
                "test": 1,
                "base_path": yuv_dir.name,
                "src_type": "yuv420",
                "sequences": sequences,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    work_root = Path(args.work_root).resolve()
    if args.existing_split:
        create_existing_split_view(args, input_root, work_root)
        return

    frame_root = work_root / "frames"
    all_dir = frame_root / "all"
    train_dir = frame_root / "train"
    val_dir = frame_root / "val"
    config_dir = work_root / "configs"
    yuv_link_dir = work_root / "yuv"

    if args.source_type in {"auto", "yuv"}:
        yuv_items = discover_yuv(input_root, args.width, args.height, args.bit_depth)
    else:
        yuv_items = []
    if args.source_type in {"auto", "png"}:
        png_items = discover_png_dirs(input_root, args.width, args.height)
    else:
        png_items = []

    source_type = "yuv" if yuv_items else "png" if png_items else ""
    if args.source_type != "auto":
        source_type = args.source_type
    seqs = yuv_items if source_type == "yuv" else png_items
    if not seqs:
        raise FileNotFoundError(f"no BVI-DVC 512 {args.source_type} sequences found under {input_root}")

    ensure_empty_or_existing(all_dir, args.overwrite)
    ensure_empty_or_existing(train_dir, args.overwrite)
    ensure_empty_or_existing(val_dir, args.overwrite)
    config_dir.mkdir(parents=True, exist_ok=True)

    if source_type == "yuv":
        yuv_link_dir.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            link_or_copy(seq["path"], yuv_link_dir / f"{seq['name']}.yuv", args.split_mode)
            frame_count = convert_yuv_to_png(seq, all_dir / seq["name"], args.bit_depth, args.max_frames)
            seq["frames"] = frame_count
    else:
        for seq in seqs:
            frame_count = standardize_png_dir(seq, all_dir / seq["name"], args.split_mode, args.max_frames)
            seq["frames"] = frame_count

    val_names = {item.strip() for item in args.val_names.split(",") if item.strip()}
    train_seqs, val_seqs = split_sequences(seqs, args.val_ratio, val_names)
    create_split_frame_dirs(all_dir, train_dir, train_seqs, args.split_mode)
    create_split_frame_dirs(all_dir, val_dir, val_seqs, args.split_mode)
    create_rgb_config(config_dir / "bvidvc512_rgb.json", frame_root, val_seqs, args.eval_frames)
    if source_type == "yuv":
        create_yuv_config(config_dir / "bvidvc512_yuv420.json", work_root, yuv_link_dir, val_seqs, args.eval_frames)

    manifest = {
        "input_root": str(input_root),
        "work_root": str(work_root),
        "source_type": source_type,
        "width": args.width,
        "height": args.height,
        "train_sequences": [s["name"] for s in train_seqs],
        "val_sequences": [s["name"] for s in val_seqs],
        "frame_root": str(frame_root),
        "rgb_config": str((config_dir / "bvidvc512_rgb.json").resolve()),
        "yuv_config": str((config_dir / "bvidvc512_yuv420.json").resolve()) if source_type == "yuv" else None,
    }
    (work_root / "manifest.json").write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    print(json.dumps(manifest, indent=4, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare BVI-DVC 512 for DCVC-FM satellite experiments")
    p.add_argument("--input_root", required=True, help="BVI-DVC 512 root containing yuv files or PNG sequence dirs")
    p.add_argument("--work_root", required=True, help="output root for frame splits and configs")
    p.add_argument("--source_type", choices=["auto", "yuv", "png"], default="auto")
    p.add_argument("--existing_split", action="store_true",
                   help="input_root already contains train/ and val/ PNG sequence folders")
    p.add_argument("--train_dir_name", type=str, default="train")
    p.add_argument("--val_dir_name", type=str, default="val")
    p.add_argument("--test_dir_name", type=str, default="",
                   help="legacy alias used only when --val_dir_name is empty")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--bit_depth", type=int, default=8)
    p.add_argument("--eval_frames", type=int, default=96, help="frames used by test_video configs; <=0 means all")
    p.add_argument("--max_frames", type=int, default=0, help="maximum frames converted/linked per sequence; <=0 means all")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--val_names", type=str, default="", help="comma-separated sequence names to force into val")
    p.add_argument("--split_mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
