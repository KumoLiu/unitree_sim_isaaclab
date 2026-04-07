"""Isaac Lab 3.0: asset/sensor .data.* properties return wp.array instead of torch.Tensor.
This module provides a helper to convert wp.array to torch.Tensor transparently."""

import warp as wp
import torch


def to_torch(data):
    """Convert wp.array to torch.Tensor if needed; pass through torch.Tensor unchanged."""
    if isinstance(data, wp.array):
        return wp.to_torch(data)
    return data
