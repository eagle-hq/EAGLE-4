# Copyright 2026 Eagle AI Corp.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available

"""Which GPU can run which kernel: the single place that reads a device's architecture.

Launchers ask this module (``clusters``, ``pdl``, ``gluon_gdn``, ...); nothing else compares compute capabilities or
device names. Importing it needs only torch, never initializes CUDA and never raises: on hardware the kernels do not
cover, :func:`unsupported_reason` says why and :func:`require_supported` raises :class:`UnsupportedHardware` with that
message, so callers can switch the package off cleanly.

======================  =========================================================================================
tier                    hardware
======================  =========================================================================================
Triton (every kernel)   NVIDIA sm_80+ (A100, H100/H200, B200/B300, consumer Blackwell sm_120), AMD CDNA2-4
Native CUDA (extras)    NVIDIA only: candidate tail from sm_80, cluster shard top-k from sm_90 (else Triton)
PDL, Gluon GDN          NVIDIA sm_90+ / the validated Gluon architectures (see :data:`GLUON_GDN_SM`)
======================  =========================================================================================

``python -m eagle_harness.kernels.arch`` prints the report for every visible device.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Union

import torch

MIN_NVIDIA_SM = 80
# AMD architectures the Triton kernels are compile-checked for (CDNA2, CDNA3, CDNA4).
AMD_ARCHS = ("gfx90a", "gfx942", "gfx950")
# Architectures the Gluon GDN tree kernel was measured faster on (every TP shape of Qwen3.8-27B); B300 only so far.
# Bit-identity is not assumed from the Triton version: gdn_tree_gluon.gluon_dispatch checks every launch signature
# bitwise against the Triton walk first.
GLUON_GDN_SM = frozenset({103})

DeviceLike = Union[None, int, str, torch.device]


class UnsupportedHardware(RuntimeError):
    """The visible GPU (or the software stack) is outside what the kernels cover."""


@dataclass(frozen=True)
class Gpu:
    """One visible GPU: index, name, vendor and architecture."""

    index: int
    name: str
    vendor: str  # "nvidia" | "amd"
    arch: str  # "sm_103" | "gfx942"
    sm: int  # CUDA compute capability as major * 10 + minor; 0 on AMD

    @property
    def nvidia(self) -> bool:
        """Whether the GPU is an NVIDIA one."""
        return self.vendor == "nvidia"


def _index(device: DeviceLike) -> Optional[int]:
    """The CUDA/HIP device index behind ``device``, None for a CPU device."""
    if isinstance(device, int):
        return device
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        return None
    return torch.cuda.current_device() if dev.index is None else dev.index


@functools.lru_cache(maxsize=None)
def _probe(index: int) -> Gpu:
    props = torch.cuda.get_device_properties(index)
    if torch.version.hip:
        return Gpu(index, props.name, "amd", props.gcnArchName.split(":")[0], 0)
    return Gpu(index, props.name, "nvidia", f"sm_{props.major}{props.minor}", props.major * 10 + props.minor)


def gpu(device: DeviceLike = None) -> Optional[Gpu]:
    """The GPU behind ``device`` (default: the current one), None without one."""
    if not torch.cuda.is_available():
        return None
    index = _index(device)
    return _probe(index) if index is not None else None


@functools.lru_cache(maxsize=None)
def _triton_error() -> Optional[str]:
    try:
        import triton  # noqa: F401
    except ImportError as e:
        return f"Triton is not importable ({e})"
    return None


def unsupported_reason(device: DeviceLike = None) -> Optional[str]:
    """None when the kernels run on ``device``, else a one-line reason."""
    g = gpu(device)
    if g is None:
        return "no CUDA or ROCm device is visible to torch"
    if (error := _triton_error()) is not None:
        return error
    if g.nvidia and g.sm < MIN_NVIDIA_SM:
        return f"{g.name} ({g.arch}) is older than NVIDIA sm_{MIN_NVIDIA_SM} (A100)"
    if not g.nvidia and g.arch not in AMD_ARCHS:
        return f"{g.name} ({g.arch}) is not one of the supported AMD architectures {', '.join(AMD_ARCHS)}"
    return None


def require_supported(device: DeviceLike = None) -> Gpu:
    """The GPU behind ``device``; raises :class:`UnsupportedHardware` with the reason where the kernels do not run."""
    if (reason := unsupported_reason(device)) is not None:
        raise UnsupportedHardware(f"eagle-harness kernels are off: {reason}.")
    return gpu(device)  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------------------------------
# Feature predicates (False on unsupported hardware; one definition each)
# ---------------------------------------------------------------------------------------------------------------------


def _nvidia_sm(device: DeviceLike, minimum: int) -> bool:
    g = gpu(device)
    return g is not None and g.nvidia and g.sm >= minimum and unsupported_reason(device) is None


def native_candidate_tail(device: DeviceLike = None) -> bool:
    """The CUDA candidate-tail kernel (warp intrinsics, sm_80+)."""
    return _nvidia_sm(device, 80)


def clusters(device: DeviceLike = None) -> bool:
    """Thread-block clusters and distributed shared memory (the CUDA shard top-k kernel), sm_90+."""
    return _nvidia_sm(device, 90)


def pdl(device: DeviceLike = None) -> bool:
    """Programmatic dependent launch (``griddepcontrol``), sm_90+ on NVIDIA."""
    return _nvidia_sm(device, 90)


def gluon_gdn(device: DeviceLike = None) -> bool:
    """The Gluon GDN tree-verify kernel may serve this device (its run-time bitwise self-check still gates it)."""
    g = gpu(device)
    return g is not None and g.nvidia and g.sm in GLUON_GDN_SM and unsupported_reason(device) is None


def _amd(device: DeviceLike) -> bool:
    g = gpu(device)
    return g is not None and not g.nvidia and unsupported_reason(device) is None


def gdn_records_replay(device: DeviceLike = None) -> bool:
    """The GDN target verify keeps per-node replay records (delta, key, decay) and replays the accepted path into
    the state pool, instead of per-node state snapshots (:mod:`eagle_harness.kernels.gdn_tree_records`). AMD CDNA:
    the snapshot kernels were measured slower there; NVIDIA keeps its snapshot walk (not measured with records)."""
    return _amd(device)


def side_stream_prelaunch(device: DeviceLike = None) -> bool:
    """The next round's draft extend and first draft steps are queued on a side stream at the end of a round
    (:meth:`eagle_harness.sglang.drafter.DynamicDrafter.prelaunch`). Measured off on AMD: MI355X TP1 bs1 rounds
    were slower with it (see the prelaunch rows in the ROCm checkpoint), so the draft stays on the caller stream."""
    g = gpu(device)
    return g is not None and g.nvidia


def report() -> str:
    """One block per visible device: the tier and every feature predicate."""
    if not torch.cuda.is_available():
        return f"no GPU visible: {unsupported_reason()}"
    lines = []
    for i in range(torch.cuda.device_count()):
        g = _probe(i)
        reason = unsupported_reason(i)
        lines.append(f"cuda:{i} {g.name} ({g.arch}): " + (f"OFF, {reason}" if reason else "supported"))
        if reason is None:
            features = (native_candidate_tail, clusters, pdl, gluon_gdn, gdn_records_replay, side_stream_prelaunch)
            lines.append("    " + ", ".join(f"{f.__name__}={'yes' if f(i) else 'no'}" for f in features))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
