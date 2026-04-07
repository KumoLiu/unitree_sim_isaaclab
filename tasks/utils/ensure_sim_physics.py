"""Isaac Lab 3.0: SimulationCfg.physics defaults to None; PhysX settings require PhysxCfg."""


def ensure_sim_has_physx_cfg(sim) -> None:
    if sim.physics is None:
        from isaaclab_physx.physics import PhysxCfg

        sim.physics = PhysxCfg()
