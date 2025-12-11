# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
import torch
project_root = os.environ.get("PROJECT_ROOT")
# usd_root = "/mnt/hdd/Data"
usd_root = "/home/nvidia/workspace/yunl/assets"
@configclass
class SurgicalSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{usd_root}/lw_v3/assets/scene02.usd",  # use simple room model
        ),
    )

    trocar_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trocar_1",
        spawn=UsdFileCfg(
            usd_path=f"{usd_root}/lw_v2/Assets/Trocar002/Trocar002_wo.usd",
            # collision properties: important for articulation interaction
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,       # increase to 0.01, important for articulation interaction
                rest_offset=-0.001,        # negative value allows slight overlap, improve stability
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-1.60202, 1.91362, 0.87183],
            # rot=[0.57824, -0.57824, 0.40699, 0.40699]
            rot=[0.0, -0.0, 0.70711, 0.70711]
        ),
    )
    
    trocar_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/trocar_2",
        spawn=UsdFileCfg(
            usd_path=f"{usd_root}/lw_v3/assets/Assets/DisposableLaparoscopicPunctureDevice001/DisposableLaparoscopicPunctureDevice005.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=[-1.48804, 1.90997, 0.86587],
            rot=[0.69692, -0.71475, -0.000243, 0.05853],
            pos=[-1.50635, 1.90997, 0.8631]
            # rot=[0.69578, -0.69533, -0.17575, -0.03896]
        ),
    )
    box = ArticulationCfg(
        prim_path="/World/envs/env_.*/box",
        spawn=UsdFileCfg(
            usd_path=f"/home/nvidia/workspace/yunl/surgery-room-dev-internal/assets/Assets/Assets/Box001/Box001.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[-1.54919, 2.03365, 0.84554],
            rot=[0.70711, 0.0, 0.0, -0.70711]
        ),
        actuators={},  # Empty dict for passive articulation (no motors)
    )
    
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=1000.0),    # light intensity
    )

