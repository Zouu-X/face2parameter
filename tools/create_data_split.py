#!/usr/bin/env python3
"""
Create reproducible train/test splits for Face2Parameter datasets.

Given a ``param.json`` mapping image identifiers to parameter vectors and a directory
containing the corresponding ``.png`` files, this script produces shuffled index files
that describe which samples belong to a split. Images are not copied; instead we store
the relative paths and reuse them inside dataloaders.

Example:
    python tools/create_data_split.py \\
        --param /data/param.json \\
        --image-root /data/images \\
        --output-dir ./splits \\
        --train-ratio 0.9
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param",
        required=True,
        help="Path to param.json containing <image_id>: [parameters] entries.",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Directory that holds the image files (filenames should match keys in param.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the split JSON files into.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of samples to assign to the train split (default: 0.9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="If set, generate a single 'train.json' containing all entries (no hold-out split).",
    )
    parser.add_argument(
        "--ext",
        default=".png",
        help="Image file extension (default: .png).",
    )
    return parser.parse_args()


def load_param_keys(param_path: str) -> List[str]:
    with open(param_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Expected dict at root of {param_path}, got {type(content).__name__}")
    return list(content.keys())


def build_paths(
    keys: Iterable[str],
    image_root: Path,
    ext: str,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return (key, relative path) tuples and missing key list."""
    pairs: List[Tuple[str, str]] = []
    missing: List[str] = []

    for key in keys:
        filename = f"{key}{ext}"
        abspath = image_root / filename
        if abspath.is_file():
            relpath = str(abspath.relative_to(image_root))
            pairs.append((key, relpath))
        else:
            missing.append(key)
    return pairs, missing


def split_pairs(
    pairs: List[Tuple[str, str]],
    train_ratio: float,
    seed: int,
    test_only: bool,
) -> Dict[str, List[Tuple[str, str]]]:
    random.seed(seed)
    random.shuffle(pairs)

    if test_only:
        return {"train": pairs}

    train_size = int(len(pairs) * train_ratio)
    train_split = pairs[:train_size]
    val_split = pairs[train_size:]
    return {"train": train_split, "val": val_split}


def save_split(output_dir: Path, split_name: str, items: List[Tuple[str, str]]) -> None:
    records = [{"key": key, "path": path} for key, path in items]
    output_path = output_dir / f"{split_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} entries to {output_path}")


def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = load_param_keys(args.param)
    pairs, missing = build_paths(keys, image_root, args.ext)

    if missing:
        print(f"WARNING: {len(missing)} keys are missing corresponding image files.")
        for sample in missing[:10]:
            print(f"  - {sample}{args.ext}")
        if len(missing) > 10:
            print("  ...")

    splits = split_pairs(pairs, args.train_ratio, args.seed, args.test_only)

    for name, items in splits.items():
        save_split(output_dir, name, items)

    if "val" in splits:
        print(f"Train/val ratio => {len(splits['train'])}:{len(splits['val'])}")


if __name__ == "__main__":
    main()
