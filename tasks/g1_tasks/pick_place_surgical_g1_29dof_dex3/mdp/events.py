# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Custom event functions for pick place surgical environment."""

from __future__ import annotations

__all__ = ["reset_box_with_random_rotation", "reset_robot_to_default_joint_positions", "object_drop_termination"]

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_box_with_random_rotation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_cfg: SceneEntityCfg,
    trocar_1_cfg: SceneEntityCfg,
    trocar_2_cfg: SceneEntityCfg,
    rotation_range: tuple[float, float] | float = (-5.0, 5.0),  # (min, max) degrees or ±value
):
    """Reset box with random rotation while keeping relative positions of trocars.
    
    This function:
    1. Applies a random yaw rotation within rotation_range to the box
    2. Rotates trocar_1 and trocar_2 around the box center to maintain relative positions
    3. Uses separate pose/velocity writes to ensure instant teleportation (no interpolation)
    
    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        box_cfg: Scene entity config for the box.
        trocar_1_cfg: Scene entity config for trocar_1.
        trocar_2_cfg: Scene entity config for trocar_2.
        rotation_range: Rotation angle range in degrees. Can be:
            - tuple (min, max): Random rotation between min and max degrees
            - float value: Random rotation between -value and +value degrees
            Examples: (0, 10), (-5, 15), 5.0 (equivalent to (-5, 5))
    """
    if len(env_ids) == 0:
        return
    
    # Parse rotation_range parameter
    if isinstance(rotation_range, (tuple, list)):
        # User provided (min, max) range
        min_angle_deg, max_angle_deg = rotation_range[0], rotation_range[1]
    else:
        # User provided single value (symmetric range ±value)
        min_angle_deg, max_angle_deg = -rotation_range, rotation_range
    
    # Get assets
    box = env.scene[box_cfg.name]
    trocar_1 = env.scene[trocar_1_cfg.name]
    trocar_2 = env.scene[trocar_2_cfg.name]
    
    # Get default states (initial positions from config)
    # 注意：default_root_state 是相对于环境原点的局部坐标
    box_default_state = box.data.default_root_state[env_ids].clone()
    trocar_1_default_state = trocar_1.data.default_root_state[env_ids].clone()
    trocar_2_default_state = trocar_2.data.default_root_state[env_ids].clone()
    
    # 关键：获取每个环境的世界坐标偏移（多环境支持）
    env_origins = env.scene.env_origins[env_ids]  # (num_envs, 3)
    
    # 将局部坐标转换为世界坐标
    box_default_state[:, :3] += env_origins
    trocar_1_default_state[:, :3] += env_origins
    trocar_2_default_state[:, :3] += env_origins
    
    # Box center position (pivot point for rotation) - 现在是世界坐标
    box_center = box_default_state[:, :3]  # (num_envs, 3)
    
    # Generate random yaw angles (in radians)
    # Convert degrees to radians
    min_angle_rad = min_angle_deg * math.pi / 180.0
    max_angle_rad = max_angle_deg * math.pi / 180.0
    
    # Generate random angles uniformly distributed in [min_angle, max_angle]
    random_yaw = torch.rand(len(env_ids), device=env.device) * (max_angle_rad - min_angle_rad) + min_angle_rad  # (num_envs,)
    
    # Create rotation quaternion for yaw (rotation around Z-axis)
    # quat = [w, x, y, z] = [cos(θ/2), 0, 0, sin(θ/2)]
    half_angle = random_yaw / 2.0
    delta_quat = torch.zeros(len(env_ids), 4, device=env.device)
    delta_quat[:, 0] = torch.cos(half_angle)  # w
    delta_quat[:, 3] = torch.sin(half_angle)  # z
    
    # Apply rotation to box quaternion
    box_new_quat = quat_multiply(delta_quat, box_default_state[:, 3:7])
    
    # Update box state
    box_new_state = box_default_state.clone()
    box_new_state[:, 3:7] = box_new_quat
    
    # Rotate trocar positions around box center
    trocar_1_relative_pos = trocar_1_default_state[:, :3] - box_center
    trocar_2_relative_pos = trocar_2_default_state[:, :3] - box_center
    
    # Rotate relative positions using the delta quaternion
    trocar_1_new_relative_pos = quat_rotate_vector(delta_quat, trocar_1_relative_pos)
    trocar_2_new_relative_pos = quat_rotate_vector(delta_quat, trocar_2_relative_pos)
    
    # New absolute positions
    trocar_1_new_state = trocar_1_default_state.clone()
    trocar_2_new_state = trocar_2_default_state.clone()
    
    trocar_1_new_state[:, :3] = box_center + trocar_1_new_relative_pos
    trocar_2_new_state[:, :3] = box_center + trocar_2_new_relative_pos
    
    # Also rotate trocar orientations
    trocar_1_new_state[:, 3:7] = quat_multiply(delta_quat, trocar_1_default_state[:, 3:7])
    trocar_2_new_state[:, 3:7] = quat_multiply(delta_quat, trocar_2_default_state[:, 3:7])
    
    zero_velocity = torch.zeros(len(env_ids), 6, device=env.device)  # [lin_vel(3), ang_vel(3)]
    
    box.write_root_pose_to_sim(box_new_state[:, :7], env_ids=env_ids)
    trocar_1.write_root_pose_to_sim(trocar_1_new_state[:, :7], env_ids=env_ids)
    trocar_2.write_root_pose_to_sim(trocar_2_new_state[:, :7], env_ids=env_ids)
    
    box.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    trocar_1.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    trocar_2.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

def object_drop_termination(
    env: ManagerBasedRLEnv,
    drop_height_threshold: float = 0.5,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
) -> torch.Tensor:
    """Termination function that triggers when objects drop below threshold.
    
    This can be used as an alternative to auto-reset, marking the episode as terminated
    so the training framework handles the reset.
    
    Args:
        env: The environment instance
        drop_height_threshold: Height below which objects are considered dropped
        asset_cfg1: Configuration for first trocar
        asset_cfg2: Configuration for second trocar
        
    Returns:
        Boolean tensor indicating which environments should terminate due to drops
    """
    # Get rigid objects
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]
    
    # Get positions
    pos1 = obj1.data.root_pos_w
    pos2 = obj2.data.root_pos_w
    
    # Check if either object has dropped
    dropped_1 = pos1[:, 2] < drop_height_threshold
    dropped_2 = pos2[:, 2] < drop_height_threshold
    
    dropped = dropped_1 | dropped_2
    
    if dropped.any():
        print(f"🔴 Drop termination triggered for {dropped.sum().item()} environment(s)")
    
    return dropped

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (Hamilton product).
    
    Quaternion format: [w, x, y, z]
    
    Args:
        q1: First quaternion (N, 4)
        q2: Second quaternion (N, 4)
        
    Returns:
        Product quaternion (N, 4)
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


def quat_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z] (N, 4)
        v: Vector to rotate (N, 3)
        
    Returns:
        Rotated vector (N, 3)
    """
    # Convert vector to quaternion [0, x, y, z]
    v_quat = torch.zeros(v.shape[0], 4, device=v.device)
    v_quat[:, 1:4] = v
    
    # q * v * q^(-1)
    # For unit quaternions, q^(-1) = q_conjugate = [w, -x, -y, -z]
    q_conj = q.clone()
    q_conj[:, 1:4] = -q_conj[:, 1:4]
    
    # Perform rotation: q * v * q_conj
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    
    return result[:, 1:4]  # Return only the vector part


def reset_robot_to_default_joint_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
):
    """Reset robot joint positions directly to default values.
    
    This function directly writes joint positions and velocities to the simulation,
    bypassing the PD controller. This prevents the "drive to target" behavior
    that causes arms to swing from 0 position to the target position.
    
    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        robot_cfg: Scene entity config for the robot.
    """
    if len(env_ids) == 0:
        return
    
    # Get robot asset
    robot = env.scene[robot_cfg.name]
    
    # Get default joint positions and velocities
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    
    # Directly write joint state to simulation (bypasses PD controller)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    
    # Also reset root state
    default_root_state = robot.data.default_root_state[env_ids].clone()
    robot.write_root_state_to_sim(default_root_state, env_ids)

