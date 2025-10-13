#!/usr/bin/env python3
"""Inspect parameter dimensionality in ``param.json``.

The module exposes :func:`inspect_param_lengths` so it can be imported and used
directly inside notebooks:

.. code-block:: python

    from tools.check_param_dimension import inspect_param_lengths
    report = inspect_param_lengths("/path/to/param.json", sample=5)
    report["unique_lengths"]

It can still be executed as a standalone script:

.. code-block:: bash

    python tools/check_param_dimension.py --param /path/to/param.json --sample 5
"""

import argparse
import json
import os
import random
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the length of continuous parameter vectors in param.json."
    )
    parser.add_argument(
        "--param",
        required=True,
        help="Path to param.json containing <image_id>: [parameter list] entries.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample entries to print for manual inspection (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when sampling keys (default: 42).",
    )
    return parser.parse_args()


def load_params(path: str) -> Dict[str, List[float]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"param json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError(f"Expected dict at root of {path}, got {type(content).__name__}")
    return content


def inspect_param_lengths(
    param_path: str,
    sample: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Inspect the parameter vector lengths inside ``param.json``.

    Args:
        param_path: Path to the JSON file mapping image ids to parameter lists.
        sample: How many entries to sample for the returned preview.
        seed: Random seed for sampling.

    Returns:
        A dictionary containing summary statistics and sampled entry metadata.
    """
    params = load_params(param_path)
    if not params:
        return {
            "total_entries": 0,
            "unique_lengths": [],
            "min_length": None,
            "max_length": None,
            "recommended": None,
            "samples": [],
        }

    lengths = [len(value) for value in params.values()]
    unique_lengths = sorted(set(lengths))
    sample_count = min(sample, len(params))
    random.seed(seed)
    sample_keys = random.sample(list(params.keys()), sample_count)

    samples = [{"key": key, "length": len(params[key])} for key in sample_keys]
    recommended = unique_lengths[0] if len(unique_lengths) == 1 else None

    return {
        "total_entries": len(params),
        "unique_lengths": unique_lengths,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "recommended": recommended,
        "samples": samples,
    }


def print_report(report: Dict[str, object]) -> None:
    """Pretty-print the report returned by :func:`inspect_param_lengths`."""
    if report["total_entries"] == 0:
        print("param.json contains no entries.")
        return

    print(f"Total entries: {report['total_entries']}")
    print(f"Unique vector lengths: {report['unique_lengths']}")
    print(f"Min length: {report['min_length']}, Max length: {report['max_length']}")
    if report["recommended"] is not None:
        print(f"Recommended continuous_params_size => {report['recommended']}")
    else:
        print("WARNING: Found multiple vector lengths. Investigate inconsistent records above.")

    print(f"\nSampled {len(report['samples'])} entries:")
    for sample in report["samples"]:
        print(f"- {sample['key']}: length={sample['length']}")


def main() -> None:
    args = parse_args()
    report = inspect_param_lengths(args.param, sample=args.sample, seed=args.seed)
    print_report(report)


if __name__ == "__main__":
    main()
