#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import torch

import hydra
from hydra import initialize, compose
from hydra.utils import instantiate

from ..utils.weight_adapter import map_training_to_inference_keys


def guess_net_alias(base_model_name: str) -> str:
    name = base_model_name.lower()
    if "large-patch14" in name:
        return "clip-vit-large-patch14"
    if "base-patch16" in name:
        return "clip-vit-base-patch16"
    # default to large
    return "clip-vit-large-patch14"


def prefix_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{k}": v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser(description="Reconstruct full model checkpoint from a slim checkpoint")
    ap.add_argument("input", type=str, help="Path to slim checkpoint (.ckpt or .pt)")
    ap.add_argument("output", type=str, help="Path to write reconstructed full checkpoint (.ckpt)")
    ap.add_argument("--base_model_name", type=str, default=None, help="Override base model name (defaults to meta)")
    ap.add_argument("--net_alias", type=str, default=None, help="Override Hydra net alias (e.g., clip-vit-large-patch14)")
    ap.add_argument("--report", type=str, default=None, help="Where to save a JSON report of mapping/load")
    args = ap.parse_args()

    ckpt_path = Path(args.input)
    out_path = Path(args.output)

    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        slim_sd = raw["state_dict"]
        meta = raw.get("meta", {})
    else:
        slim_sd = raw
        meta = {}

    base_model_name = args.base_model_name or meta.get("base_model_name") or "openai/clip-vit-large-patch14"
    net_alias = args.net_alias or guess_net_alias(base_model_name)

    # Compose Hydra config and instantiate the base ReuseModel
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"model/net={net_alias}",
                "trainer=cpu",
                "logger=csv",
            ],
        )
    model = instantiate(cfg.model.net)

    # Load slim weights into the model (strip Lightning prefixes)
    remapped = map_training_to_inference_keys(slim_sd)
    load_res = model.load_state_dict(remapped, strict=False)

    # Build a Lightning-style state_dict by prefixing with 'net.'
    full_model_sd = model.state_dict()
    full_lit_sd = prefix_state(full_model_sd, "net.")

    recon = {
        "state_dict": full_lit_sd,
        "meta": {
            **meta,
            "reconstructed_from_slim": True,
            "base_model_name": base_model_name,
            "net_alias": net_alias,
        },
    }
    torch.save(recon, out_path)

    if args.report:
        report = {
            "missing_keys": load_res.missing_keys,
            "unexpected_keys": load_res.unexpected_keys,
            "total_params": sum(p.numel() for p in model.parameters()),
        }
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"Reconstructed checkpoint written to: {out_path}")


if __name__ == "__main__":
    sys.exit(main())

