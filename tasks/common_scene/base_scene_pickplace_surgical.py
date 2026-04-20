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
      # 1. room wall configuration - disable collisions to avoid GPU-cloth incompatibility
    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path="/home/nvidia/workspace/mingxue/surgery-room-dev-internal/assets/Assets/scene04.usd",
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
    )

    # # Cloth (deformable object)
    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        spawn=UsdFileCfg(
            usd_path="/home/nvidia/workspace/mingxue/surgery-room-dev-internal/assets/Assets/Assets/Cloth/Cloth_fold03/Cloth_fold04.usd",
            physics_material=SurfaceDeformableBodyMaterialCfg(
                density=200.0,
                youngs_modulus=5e5,
                poissons_ratio=0.1,
                surface_stretch_stiffness=1.0,
                surface_shear_stiffness=5000.0,
                surface_bend_stiffness=5.0,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(-1.51, 2.505, 0.79),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # cloth: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Cloth",
    #     spawn=UsdFileCfg(
    #         usd_path="/home/nvidia/workspace/mingxue/surgery-room-dev-internal/assets/Assets/Assets/Cloth/Cloth_fold03/Cloth_fold03.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(-1.51, 2.355, 0.80),
    #         rot=(0.0, 0.0, 0.0, 1.0),
    #     ),
    # )

    # Ground plane
    # # 3. ground configuration
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",    # ground in the scene
    #     spawn=GroundPlaneCfg( ),    # ground configuration
    # )

    # table_top_collider = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TableTopCollider",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.8, 0.6, 0.02),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(1.0, 1.0, 1.0),
    #             opacity=0.0,
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(-1.51, 2.355, 0.77),
    #         rot=(0.0, 0.0, 0.70711, 0.70711),
    #     ),
    # )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=1000.0),    # light intensity
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-0.1, 3.6, 1.6),
                                                    rot_offset=(0.00617, 0.70708, -0.70708, -0.00617),
                                                    focal_length = 16.5)