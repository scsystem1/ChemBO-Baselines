from __future__ import annotations

import os

import torch


def resolve_device_type(preferred: str | None = None) -> str:
    requested_device = (preferred or os.getenv("GOLLUM_DEVICE", "")).strip().lower()
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_device(preferred: str | None = None) -> torch.device:
    return torch.device(resolve_device_type(preferred))


def empty_cuda_cache(preferred: str | None = None) -> None:
    if resolve_device_type(preferred) == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
