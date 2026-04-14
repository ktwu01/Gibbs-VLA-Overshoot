"""Simplified VLA wrappers for trajectory extraction."""

from .vla_wrapper import (
    InstructionSwapRunner,
    MockPolicy,
    PolicyAdapter,
    SwapResult,
    TrajectoryBatch,
    VLATrajectoryExtractor,
)

__all__ = [
    "InstructionSwapRunner",
    "MockPolicy",
    "PolicyAdapter",
    "SwapResult",
    "TrajectoryBatch",
    "VLATrajectoryExtractor",
]

# Lazy import: OpenVLAPolicy requires torch/transformers
def __getattr__(name: str):
    if name == "OpenVLAPolicy":
        from .openvla_policy import OpenVLAPolicy
        return OpenVLAPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
