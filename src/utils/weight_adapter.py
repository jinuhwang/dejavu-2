from __future__ import annotations

from typing import Dict, Tuple, List

import torch


def _apply_simple_key_rules(key: str) -> str:
    """
    Apply common rename rules seen across training/inference checkpoints.

    Rules are conservative and order-sensitive.
    """
    rules: List[Tuple[str, str]] = [
        # Older checkpoints sometimes include extra nesting prefixes
        ("model.base_model.model.", "model."),
        ("base_model.model.", "model."),
        ("net.", ""),
    ]
    for src, dst in rules:
        if key.startswith(src):
            key = key.replace(src, dst, 1)
    return key


def map_training_to_inference_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Return a new state_dict with keys adapted for loading into the inference model.

    The mapping is currently identity for most parameters because the inference
    wrapper reuses the same underlying modules. However, we normalize a few
    common prefixes to reduce friction when loading legacy checkpoints.
    """
    remapped: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = _apply_simple_key_rules(k)
        remapped[new_k] = v
    return remapped


def load_with_report(module: torch.nn.Module, ckpt: Dict[str, torch.Tensor], strict: bool = False):
    """
    Load 'ckpt' into 'module' and return a report dictionary with matched,
    renamed (changed keys), missing, and unexpected.
    """
    # Detect potential renames
    renamed: List[Tuple[str, str]] = []
    for k in ckpt.keys():
        new_k = _apply_simple_key_rules(k)
        if new_k != k:
            renamed.append((k, new_k))

    remapped = map_training_to_inference_keys(ckpt)
    missing_before = set(module.state_dict().keys()) - set(remapped.keys())
    res = module.load_state_dict(remapped, strict=strict)
    missing_after = set(res.missing_keys)
    unexpected = set(res.unexpected_keys)

    matched = set(remapped.keys()) - unexpected
    report = {
        "matched": sorted(matched),
        "renamed": renamed,
        "missing": sorted(missing_after),
        "unexpected": sorted(unexpected),
    }
    return report, res
