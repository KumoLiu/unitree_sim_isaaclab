# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")
@configclass
class SurgicalSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    #   # 1. room wall configuration - disable collisions to avoid GPU-cloth incompatibility
    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path="/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/scene04.usd",
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            scale=(1.0, 1.0, 1.1),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.1, 0.0],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
    )

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        spawn=UsdFileCfg(
            usd_path="/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/Assets/Cloth/Cloth_fold06/Cloth_fold10.usd",
            scale=(1.0, 1.0, 1.0),
            # deformable_props=DeformableBodyPropertiesCfg(
            #     disable_gravity=False,
            # ),
            # physics_material=SurfaceDeformableBodyMaterialCfg(
            #     density=100.0,
            #     youngs_modulus=5e5,
            #     poissons_ratio=0.1,
            #     surface_stretch_stiffness=1.0,
            #     surface_shear_stiffness=5000,
            #     surface_bend_stiffness=5,
            # ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(-1.60, 2.505, 0.90),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=1000.0),    # light intensity
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-0.1, 3.6, 1.6),
                                                    rot_offset=(0.00617, 0.70708, -0.70708, -0.00617),
                                                    focal_length = 16.5)