"""Hardware / device description for training hosts.

``torch`` is imported lazily inside :func:`describe_device` so this module
(and everything that imports it) works on machines without torch.
"""

from __future__ import annotations


def describe_device() -> dict:
    """Describe the available compute device.

    Returns ``{"torch_available": bool, "torch_version": str | None,
    "cuda_available": bool, "device": "cuda" | "cpu", "device_count": int}``.
    Never raises for a missing torch installation.
    """
    try:
        import torch  # lazy: torch is optional
    except ImportError:
        return {
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "device": "cpu",
            "device_count": 0,
        }
    cuda_available = bool(torch.cuda.is_available())
    try:
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception:
        device_count = 0
    return {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "device_count": device_count,
    }
