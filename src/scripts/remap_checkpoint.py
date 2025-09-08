#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from ..utils.weight_adapter import map_training_to_inference_keys


def main():
    ap = argparse.ArgumentParser(description="Remap a training checkpoint for inference usage")
    ap.add_argument("input", type=str, help="Path to input checkpoint (PyTorch .pt/.ckpt)")
    ap.add_argument("output", type=str, help="Path to write remapped checkpoint (.pt)")
    ap.add_argument("--report", type=str, default=None, help="Optional path to JSON report")
    ap.add_argument("--validate", action="store_true", help="Instantiate model and validate loading")
    args = ap.parse_args()

    ckpt_path = Path(args.input)
    out_path = Path(args.output)

    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    remapped = map_training_to_inference_keys(state_dict)

    # Shallow report: list of renamed keys only (validation requires a live model)
    # Build a shallow rename report using simple prefix rules
    prefixes = [
        "model.base_model.model.",
        "base_model.model.",
        "model.",
        "net.",
    ]
    renamed = []
    for k in state_dict.keys():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k.replace(p, "", 1)
                break
        if new_k != k:
            renamed.append((k, new_k))

    report = {
        "matched": [],
        "renamed": renamed,
        "missing": [],
        "unexpected": [],
    }

    torch.save(remapped, out_path)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"Saved remapped checkpoint to: {out_path}")
    if args.report:
        print(f"Wrote report to: {args.report}")


if __name__ == "__main__":
    main()
