"""Reset the inner rigid body (Cloth_In) embedded inside the deformable cloth USD."""
from __future__ import annotations

import re
import sys
import torch
from pxr import Gf, UsdGeom


_DEFAULT_CLOTH_IN_REL_PATH = "Cloth_In001/Cloth_In001"
# Attribute names of caches we stash on the env instance (one helper, many resets):
_CACHE_INIT = "_cloth_inner_init"          # dict[env_id -> {pos_gf, rot_gf, pose7}]
_CACHE_VIEW = "_cloth_inner_rigid_view"    # PhysX rigid body view (created once)
_CACHE_GLOB = "_cloth_inner_view_pattern"  # glob string the view was built for


def _resolve_env_prim_path(prim_path_expr: str, env_id: int) -> str:
    path = prim_path_expr
    if "{ENV_REGEX_NS}" in path:
        path = path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
    path = re.sub(r"env_\.\*", f"env_{env_id}", path)
    path = re.sub(r"env_\d+", f"env_{env_id}", path)
    return path


def _expr_to_glob(prim_path_expr: str) -> str:
    """Convert IsaacLab regex prim path into a wildcard glob accepted by PhysX views."""
    g = prim_path_expr.replace("{ENV_REGEX_NS}", "/World/envs/env_*")
    g = g.replace("env_.*", "env_*")
    return g


def _get_isaaclab_sim_view():
    """Reuse the warp-backed SimulationView IsaacLab already created.

    Available after the first physics warmup (``PhysicsManager._on_play``).
    Returns ``None`` if the view does not exist yet -- callers should fall back to USD.
    """
    try:
        from isaaclab_physx.physics.physx_manager import PhysxManager
        return PhysxManager._view
    except Exception:
        return None


def _try_physx_tensor_reset(env, env_ids, prim_path_expr, inner_rel_path):
    """Reset the Cloth_In rigid bodies via the PhysX tensor (warp) API.

    Returns True on success, False so the caller can fall back to USD.
    """
    try:
        import warp as wp
        import numpy as np
    except Exception as e:
        print(f"[cloth_inner_reset] warp/numpy unavailable ({e}), USD fallback")
        return False

    sim_view = _get_isaaclab_sim_view()
    if sim_view is None:
        # Sim view isn't ready yet (very first reset before play). USD fallback is fine here
        # because the simulation hasn't started -- USD writes still propagate to PhysX.
        return False

    glob = f"{_expr_to_glob(prim_path_expr)}/{inner_rel_path}"

    # Cache the rigid body view so we don't pay the create cost on every reset
    rigid_view = getattr(env, _CACHE_VIEW, None)
    if rigid_view is None or getattr(env, _CACHE_GLOB, None) != glob:
        try:
            rigid_view = sim_view.create_rigid_body_view(glob)
        except Exception as e:
            print(f"[cloth_inner_reset] create_rigid_body_view failed ({e}), USD fallback")
            return False
        if rigid_view is None or rigid_view.count == 0:
            print(f"[cloth_inner_reset] tensor view empty for {glob}, USD fallback")
            return False
        setattr(env, _CACHE_VIEW, rigid_view)
        setattr(env, _CACHE_GLOB, glob)

    n_total = rigid_view.count
    init_dict = getattr(env, _CACHE_INIT)

    # Build pose buffer for the indices we want to reset (m x 7)
    m = len(env_ids)
    pose_np = np.empty((m, 7), dtype=np.float32)
    for i, eid in enumerate(env_ids):
        cache = init_dict.get(eid)
        if cache is None:
            print(f"[cloth_inner_reset] no cached init pose for env {eid}, USD fallback")
            return False
        p = cache["pose7"]
        pose_np[i] = p.cpu().numpy() if hasattr(p, "cpu") else p

    # Filter out env_ids beyond the view (shouldn't happen, but be safe)
    idx_np = np.array([e for e in env_ids if 0 <= int(e) < n_total], dtype=np.int32)
    if idx_np.shape[0] != m:
        print(f"[cloth_inner_reset] env_ids out of view range ({m} -> {idx_np.shape[0]}), USD fallback")
        return False

    device = str(env.device)
    try:
        pose_wp = wp.array(pose_np, dtype=wp.float32, device=device)
        idx_wp = wp.array(idx_np, dtype=wp.int32, device=device)
        vel_wp = wp.zeros((m, 6), dtype=wp.float32, device=device)
        rigid_view.set_transforms(pose_wp, indices=idx_wp)
        rigid_view.set_velocities(vel_wp, indices=idx_wp)
        return True
    except Exception as e:
        print(f"[cloth_inner_reset] tensor reset failed ({e}), invalidating view, USD fallback")
        # Drop the cached view so it gets recreated on the next attempt
        try:
            setattr(env, _CACHE_VIEW, None)
            setattr(env, _CACHE_GLOB, None)
        except Exception:
            pass
        return False


def _usd_pose_reset(stage, prim_path, init_pos, init_rot):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(init_pos)
    xformable.AddOrientOp().Set(Gf.Quatf(
        float(init_rot.GetReal()),
        Gf.Vec3f(*[float(x) for x in init_rot.GetImaginary()]),
    ))
    return True


def _autodetect_inner_rel_path(stage, prim_path_expr) -> str | None:
    """Walk the env_0 cloth template and return the rel path of the first
    descendant that carries ``PhysicsRigidBodyAPI``. This avoids hard-coding
    ``Cloth_In001`` vs ``Cloth_In002`` etc. across different fold variants.

    Logs the children it inspects so that when detection fails we can see
    exactly what hierarchy the live stage exposes (often different from the
    standalone-pxr view because of references / ``proto_asset_0`` wrappers).
    """
    base = prim_path_expr
    if "{ENV_REGEX_NS}" in base:
        base = base.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    base = re.sub(r"env_\.\*", "env_0", base)
    template = stage.GetPrimAtPath(base)
    if not template or not template.IsValid():
        print(f"[cloth_inner_reset] autodetect: template prim {base!r} is invalid")
        return None

    base_len = len(base.rstrip("/")) + 1
    visited = []
    # Use Usd.PrimRange so we traverse into instance proxies / references too,
    # not just direct GetAllChildren which can stop at instance boundaries.
    from pxr import Usd  # local import: pxr already pulled in at module load
    for prim in Usd.PrimRange.AllPrims(template):
        if prim == template:
            continue
        path_str = prim.GetPath().pathString
        applied = list(prim.GetAppliedSchemas())
        visited.append((path_str, applied))
        if "PhysicsRigidBodyAPI" in applied:
            rel = path_str[base_len:]
            print(f"[cloth_inner_reset] autodetect: found rigid body at {path_str!r}, "
                  f"rel={rel!r}")
            return rel

    print(f"[cloth_inner_reset] autodetect FAILED under {base!r}. "
          f"Visited {len(visited)} prims, none had PhysicsRigidBodyAPI:")
    for p, schemas in visited[:30]:
        print(f"   - {p}  schemas={schemas}")
    if len(visited) > 30:
        print(f"   ... and {len(visited) - 30} more")
    return None


def reset_cloth_inner(env, env_ids, cloth_asset_name: str = "cloth", inner_rel_path: str | None = None):
    """Reset Cloth_In rigid body back to its initial pose.

    Args:
        cloth_asset_name: name of the cloth attribute on ``env.scene.cfg``.
        inner_rel_path: relative path from the cloth spawn root to the inner
            rigid body (e.g. ``"Cloth_In002/Cloth_In002"``). When ``None``
            (default) the function walks the spawned cloth and picks the first
            child prim with ``PhysicsRigidBodyAPI`` applied; if auto-detection
            also fails, falls back to ``Cloth_In001/Cloth_In001`` for backward
            compatibility with older fold04/05/06 USDs.

    NOTE: ``env_ids`` is intentionally a required positional (no default).
    IsaacLab's ``EventManager._resolve_common_term_cfg(min_argc=2)``
    expects the first two parameters (``env`` and ``env_ids``) to be
    *required* -- giving ``env_ids`` a default pushes it into the validator's
    "optional" bucket and produces a misleading
    ``expects mandatory parameters: [] ... but received: ['cloth_asset_name']``
    error.  Callers that invoke this function directly (not via EventManager)
    should pass an explicit list / tensor of env indices, or
    ``list(range(env.num_envs))`` to act on all envs.
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    cloth_cfg = getattr(env.scene.cfg, cloth_asset_name, None)
    if cloth_cfg is None:
        return
    prim_path_expr = cloth_cfg.prim_path

    # Resolve which sub-prim to reset. Cache the resolved value on env to avoid
    # re-walking the USD on every reset.
    if inner_rel_path is None:
        inner_rel_path = getattr(env, "_cloth_inner_rel_path", None)
        if inner_rel_path is None:
            detected = _autodetect_inner_rel_path(stage, prim_path_expr)
            inner_rel_path = detected or _DEFAULT_CLOTH_IN_REL_PATH
            setattr(env, "_cloth_inner_rel_path", inner_rel_path)
            print(f"[cloth_inner_reset] using inner_rel_path={inner_rel_path!r} "
                  f"(autodetected={'yes' if detected else 'no, fallback to default'})")
            sys.stdout.flush()

    if env_ids is None:
        env_ids = list(range(env.num_envs))
    elif isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.detach().cpu().tolist()
    else:
        env_ids = list(env_ids)

    if not hasattr(env, _CACHE_INIT):
        setattr(env, _CACHE_INIT, {})
    init_dict = getattr(env, _CACHE_INIT)

    # First pass: make sure every requested env has a cached init pose.
    valid_env_ids = []
    inner_paths = []
    for env_id in env_ids:
        base = _resolve_env_prim_path(prim_path_expr, env_id)
        path = f"{base}/{inner_rel_path}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[cloth_inner_reset] prim not found: {path}")
            continue

        if env_id not in init_dict:
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            t = xf.ExtractTranslation()
            q = xf.ExtractRotationQuat()
            init_dict[env_id] = {
                "pos_gf": Gf.Vec3d(t),
                "rot_gf": Gf.Quatd(q),
                # PhysX tensor layout for set_transforms: [x,y,z, qx,qy,qz,qw]
                # (xyzw, NOT IsaacLab's external wxyz convention). IsaacLab
                # itself does torch.roll(quat, -1) before calling set_transforms,
                # so we just bake the xyzw order in here once.
                "pose7": torch.tensor([
                    float(t[0]), float(t[1]), float(t[2]),
                    float(q.GetImaginary()[0]),
                    float(q.GetImaginary()[1]),
                    float(q.GetImaginary()[2]),
                    float(q.GetReal()),
                ], dtype=torch.float32),
            }
            print(f"[cloth_inner_reset] cached env {env_id} init pos: {t}")

        valid_env_ids.append(env_id)
        inner_paths.append(path)

    if not valid_env_ids:
        return

    # Path A: PhysX tensor view (works on GPU + suppressReadback=True).
    if _try_physx_tensor_reset(env, valid_env_ids, prim_path_expr, inner_rel_path):
        return

    # Path B: USD xform writes + flush.
    # Only effective at startup before play, or when suppressReadback is False.
    for env_id, path in zip(valid_env_ids, inner_paths):
        cache = init_dict.get(env_id)
        if cache is None:
            continue
        _usd_pose_reset(stage, path, cache["pos_gf"], cache["rot_gf"])

    try:
        import omni.physx
        omni.physx.get_physx_interface().update_transformations(
            updateToFastCache=True, updateToUsd=False,
            updateVelocitiesToUsd=False, outputVelocitiesLocalSpace=False,
        )
    except Exception:
        pass
