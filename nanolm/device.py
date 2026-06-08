"""
Device and system detection utilities.
Single source of truth for device selection, dtype selection, and backend detection.
"""

import torch
import torch.distributed as dist


def get_optimal_device() -> str:
    """Auto-select the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_bf16_supported() -> bool:
    """Check if bfloat16 is supported on the current device."""
    if torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported'):
            return torch.cuda.is_bf16_supported()
        try:
            device_cap = torch.cuda.get_device_capability()
            return device_cap[0] >= 8
        except Exception:
            return False

    if hasattr(torch, 'npu') and torch.npu.is_available():
        return True

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return False

    return False


def is_fp16_supported() -> bool:
    return torch.cuda.is_available() or (hasattr(torch, 'npu') and torch.npu.is_available())


def get_autocast_dtype() -> torch.dtype:
    """Return the best autocast dtype for the current device."""
    return torch.bfloat16 if is_bf16_supported() else torch.float16


def get_optimal_backend() -> str:
    """Get optimal distributed backend: nccl > hccl > gloo."""
    if torch.cuda.is_available() and dist.is_nccl_available():
        return 'nccl'
    if hasattr(dist, 'is_hccl_available') and dist.is_hccl_available():
        return 'hccl'
    return 'gloo'


def resolve_device_type(device) -> str:
    """Resolve a device/string to a device_type string safe for torch.autocast.
    MPS is treated as 'cpu' because torch.autocast doesn't support MPS.
    """
    if isinstance(device, torch.device):
        dt = device.type
    elif isinstance(device, str):
        dt = device
    else:
        dt = str(device)
    return dt if dt != "mps" else "cpu"


def empty_cache():
    """Safely clear CUDA cache (no-op if CUDA unavailable)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
