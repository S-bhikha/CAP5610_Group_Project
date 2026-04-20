#!/usr/bin/env python3
"""Augment training MRI images via rotations and flips into Enhanced_Training/.

Layout matches Testing/: Enhanced_Training/<class>/*.png|jpg|jpeg
Source defaults to Training/ with the same per-class subfolders.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# (suffix, transform) — applied to a loaded RGB/L copy of the source image
AugmentFn = Callable[[Image.Image], Image.Image]


def _build_augmentations(
    rotate_angles: List[int] | None,
    rotate_step: int | None,
    enable_flips: bool,
) -> List[Tuple[str, AugmentFn]]:
    angles: List[int]
    if rotate_angles is not None and len(rotate_angles) > 0:
        angles = rotate_angles
    elif rotate_step is not None:
        if rotate_step <= 0 or rotate_step >= 360:
            raise ValueError("--rotate-step must be in the range 1..359")
        angles = list(range(rotate_step, 360, rotate_step))
    else:
        angles = [90, 180, 270]

    # Keep angles valid and avoid duplicates while preserving order
    seen = set()
    angles = [a % 360 for a in angles if (a % 360) != 0 and (a % 360) not in seen and not seen.add(a % 360)]

    augs: List[Tuple[str, AugmentFn]] = []

    for a in angles:
        if a in (90, 180, 270):
            transposed = {
                90: Image.Transpose.ROTATE_90,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_270,
            }[a]
            augs.append((f"rot{a}", lambda im, t=transposed: im.transpose(t)))
        else:
            augs.append(
                (
                    f"rot{a}",
                    lambda im, deg=a: im.rotate(
                        deg,
                        resample=Image.Resampling.BILINEAR,
                        expand=False,
                        fillcolor=0,
                    ),
                )
            )

    if enable_flips:
        augs.extend(
            [
                ("flip_lr", lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
                ("flip_tb", lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
            ]
        )

    return augs


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def augment_class_folder(
    src_class_dir: Path,
    dst_class_dir: Path,
    augmentations: List[Tuple[str, AugmentFn]],
    exif_transpose: bool,
) -> Tuple[int, int]:
    """Copy originals and write augmented variants. Returns (n_sources, n_written)."""
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    n_sources = 0
    n_written = 0

    for entry in sorted(src_class_dir.iterdir()):
        if not _is_image_file(entry):
            continue
        n_sources += 1

        with Image.open(entry) as img:
            if exif_transpose:
                img = ImageOps.exif_transpose(img)
            base = img.convert("RGB") if img.mode not in ("RGB", "L") else img.copy()
            if base.mode == "L":
                work = base
            else:
                work = base.convert("L")

        # Original (same basename as Training)
        out_orig = dst_class_dir / entry.name
        work.save(out_orig)
        n_written += 1

        stem, suf = entry.stem, entry.suffix.lower() or ".png"
        for suffix, fn in augmentations:
            aug = fn(work.copy())
            out_path = dst_class_dir / f"{stem}_{suffix}{suf}"
            aug.save(out_path)
            n_written += 1

    return n_sources, n_written


def run_augmentation(
    data_root: Path,
    source_name: str,
    dest_name: str,
    classes: List[str] | None,
    exif_transpose: bool,
    rotate_angles: List[int] | None,
    rotate_step: int | None,
    no_flips: bool,
    dry_run: bool,
) -> None:
    src_root = data_root / source_name
    dst_root = data_root / dest_name

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    if classes is None:
        classes = sorted(
            d.name
            for d in src_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    augmentations = _build_augmentations(
        rotate_angles=rotate_angles,
        rotate_step=rotate_step,
        enable_flips=not no_flips,
    )
    total_src = total_out = 0

    for cls in classes:
        src_dir = src_root / cls
        dst_dir = dst_root / cls
        if not src_dir.is_dir():
            print(f"Warning: skip missing class folder -> {src_dir}")
            continue

        if dry_run:
            n = sum(1 for p in src_dir.iterdir() if _is_image_file(p))
            would = n * (1 + len(augmentations))
            print(f"[dry-run] {cls}: {n} images -> {would} files -> {dst_dir}")
            total_src += n
            total_out += would
            continue

        n_src, n_w = augment_class_folder(src_dir, dst_dir, augmentations, exif_transpose)
        print(f"{cls}: {n_src} source images -> {n_w} files written -> {dst_dir}")
        total_src += n_src
        total_out += n_w

    if dry_run:
        print(f"Total: {total_src} source images -> {total_out} files (dry-run)")
    else:
        print(f"Total: {total_src} source images -> {total_out} files written under {dst_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment Training/ images into Enhanced_Training/ (Testing-style layout)."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Folder that contains Training/, Testing/, etc. (default: current directory)",
    )
    p.add_argument(
        "--source",
        default="Training",
        help="Subfolder under data-root to read from (default: Training)",
    )
    p.add_argument(
        "--dest",
        default="Enhanced_Training",
        help="Subfolder under data-root to write (default: Enhanced_Training)",
    )
    p.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional explicit class subfolder names (default: all subfolders of source)",
    )
    p.add_argument(
        "--rotate-angles",
        nargs="*",
        type=int,
        default=None,
        help="Explicit rotation angles in degrees (e.g. --rotate-angles 1 2 3). Default: 90 180 270.",
    )
    p.add_argument(
        "--rotate-step",
        type=int,
        default=None,
        help="Generate rotations every N degrees up to 359 (e.g. --rotate-step 1). Overrides default rotations unless --rotate-angles is set.",
    )
    p.add_argument(
        "--no-flips",
        action="store_true",
        help="Disable mirror flips (left-right and top-bottom).",
    )
    p.add_argument(
        "--exif-transpose",
        action="store_true",
        help="Apply EXIF orientation before augmenting (usually off for MRI exports)",
    )
    p.add_argument(
        "--clear-dest",
        action="store_true",
        help="Remove destination folder before writing (use to avoid stale files)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()

    dst = data_root / args.dest
    if args.clear_dest and dst.exists() and not args.dry_run:
        shutil.rmtree(dst)
        print(f"Removed existing {dst}")

    run_augmentation(
        data_root=data_root,
        source_name=args.source,
        dest_name=args.dest,
        classes=args.classes,
        exif_transpose=args.exif_transpose,
        rotate_angles=args.rotate_angles,
        rotate_step=args.rotate_step,
        no_flips=args.no_flips,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
