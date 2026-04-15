# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import tempfile
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
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
from tasks.common_scene.base_scene_pickplace_surgical import SurgicalSceneCfg
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
    
    # G1 29-dof robot with Inspire hand (Nucleus USD, same as tablecloth task)
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.92, 2.5, 0.81168),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
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
    pass


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
        self.episode_length_s = 20.0
        self.sim.dt = 1/200
        self.sim.render_interval = self.decimation
        self.sim.physics.bounce_threshold_velocity = 0.01
        self.sim.physics.gpu_max_deformable_surface_contacts = 1024 * 1024 * 12
        self.sim.physics.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physics.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }

        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))
