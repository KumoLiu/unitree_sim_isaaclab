
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
from tasks.common_termination.base_termination_pick_place_cylinder import reset_object_estimate
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from tasks.utils.warp_utils import to_torch

__all__ = [
"reset_object_estimate",
"object_moved"
]

def object_moved(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.07,
) -> torch.Tensor:
    """Terminate if the object has moved by a certain distance from its initial position."""
    obj: RigidObject = env.scene[object_cfg.name]
    initial_pos_key = f"initial_pos_{object_cfg.name}"
    if initial_pos_key not in env.extras:
        env.extras[initial_pos_key] = to_torch(obj.data.root_pos_w).clone()
    is_first_step = env.episode_length_buf == 0
    if torch.any(is_first_step):
        env.extras[initial_pos_key][is_first_step] = to_torch(obj.data.root_pos_w)[is_first_step].clone()
    initial_pos = env.extras[initial_pos_key]
    current_pos = to_torch(obj.data.root_pos_w)
    distance_moved = torch.norm(current_pos[:, :3] - initial_pos[:, :3], p=2, dim=1)
    return distance_moved > distance_threshold
