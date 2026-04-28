# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_scene.base_scene_pickplace_surgical import SurgicalSceneCfg
from tasks.common_event.cloth_inner_reset import reset_cloth_inner  # isort: skip
from tasks.utils.ensure_sim_physics import ensure_sim_has_physx_cfg  # isort: skip

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(SurgicalSceneCfg):
    """object table scene configuration class
    
    inherits from SurgicalSceneCfg, gets the complete surgical scene configuration
    uses G1 29-dof with Inspire hand
    """
    
    # G1 29-dof robot with Inspire hand (base fixed, waist locked, arms raised)
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-1.92, 2.5, 0.81168),
        init_rot=(0.0, 0.0, 0.0, 1.0),
        custom_joint_pos={
            "left_shoulder_pitch_joint": -0.8,
            "right_shoulder_pitch_joint": -0.8,
            "left_shoulder_roll_joint": 0.5,
            "right_shoulder_roll_joint": -0.5,
            "left_elbow_joint": -0.3,
            "right_elbow_joint": -0.3,
        },
    )
    # camera configuration (Inspire wrist cameras)
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        """

        # 1. robot joint state observation
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        # 2. inspire hand joint state observation 
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)

        # 3. camera image observation
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

def dummy_reward(env):
    """Simple dummy reward function that returns zero for all environments"""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

@configclass
class RewardsCfg:
    pass
    reward = RewTerm(func=dummy_reward,weight=1.0)

@configclass
class EventCfg:
    """Reset all scene entities (robot, rigid objects, deformables) to their initial state."""

    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    # Cloth_In001 is a rigid body embedded inside the cloth USD, not registered as
    # a separate scene asset, so reset_scene_to_default does not touch it. Reset it
    # explicitly back to its captured init pose (and zero velocity).
    reset_cloth_inner = EventTermCfg(
        func=reset_cloth_inner,
        mode="reset",
        params={"cloth_asset_name": "cloth"},
    )


@configclass
class PickPlaceG129InspireJointEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29-dof robot with Inspire hand – surgical pick-place environment"""

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1,
                                                     env_spacing=2.5,
                                                     replicate_physics=False
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None
    def __post_init__(self):
        """Post initialization."""
        ensure_sim_has_physx_cfg(self.sim)
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation 
        self.sim.physics.bounce_threshold_velocity = 0.01
        self.sim.physics.gpu_max_deformable_surface_contacts = 2**25
