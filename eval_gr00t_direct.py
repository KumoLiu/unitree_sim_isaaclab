# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct evaluation of Gr00t policy in IsaacLab simulation
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Evaluate Gr00t Policy in IsaacLab")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Surgical-G129-Dex3-Joint-Box", help="task name")
parser.add_argument("--model_path", type=str, default=None, help="path to Gr00t model checkpoint")
parser.add_argument("--num_episodes", type=int, default=10, help="number of evaluation episodes")
parser.add_argument("--max_steps", type=int, default=500, help="max steps per episode")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--save_video", action="store_true", help="save video of evaluation")
parser.add_argument("--video_dir", type=str, default="./eval_videos", help="directory to save videos")
parser.add_argument("--grid_layout", type=str, default="4x5", help="grid layout (e.g., '4x5' for 4 rows and 5 columns)")
parser.add_argument(
    "--action_chunk_size", type=int, default=1, help="number of actions to use from action chunk (1-16, default: 1)"
)
parser.add_argument(
    "--frequency",
    type=float,
    default=0.0,
    help="control frequency (Hz) for stepping actions; emulates real-time control loop",
)
parser.add_argument("--task_description", type=str, default="install trocar from box", help="task description")
parser.add_argument(
    "--test",
    action="store_true",
    help="run a lightweight integration test with a dummy policy (still starts sim + env)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import tasks
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg



import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch


def set_viewport_camera(camera_prim_path: str) -> None:
    """Set the active viewport camera to a specific camera prim."""
    try:
        import omni.kit.commands
        from omni.kit.viewport.utility import get_active_viewport

        viewport_api = get_active_viewport()
        if viewport_api is None:
            return

        omni.kit.commands.execute(
            "SetViewportCamera",
            camera_path=camera_prim_path,
            viewport_api=viewport_api,
        )
    except Exception as e:
        print(f"⚠️ Failed to set viewport camera: {e}")
        traceback.print_exc()


def process_observation(obs: Dict[str, Any], env, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Convert IsaacLab observations to Gr00t policy input format."""
    processed_obs: Dict[str, torch.Tensor] = {}

    # Get camera images directly from scene sensors
    try:
        sensors_dict = getattr(env.scene, "sensors", None)
        if sensors_dict:
            # Front camera
            if "front_camera" in sensors_dict:
                try:
                    front_cam = sensors_dict["front_camera"]
                    front_rgb = front_cam.data.output["rgb"][0]  # (H, W, 3)
                    if front_rgb.dtype != torch.uint8:
                        front_rgb = (front_rgb * 255).to(torch.uint8)
                    processed_obs["video.room_view"] = front_rgb.unsqueeze(0).to(device)
                except Exception as e:
                    print(f"⚠️ Failed to get front camera: {e}")

            # Left wrist camera
            if "left_wrist_camera" in sensors_dict:
                try:
                    left_cam = sensors_dict["left_wrist_camera"]
                    left_rgb = left_cam.data.output["rgb"][0]
                    if left_rgb.dtype != torch.uint8:
                        left_rgb = (left_rgb * 255).to(torch.uint8)
                    processed_obs["video.left_wrist_view"] = left_rgb.unsqueeze(0).to(device)
                except Exception as e:
                    print(f"⚠️ Failed to get left wrist camera: {e}")

            # Right wrist camera
            if "right_wrist_camera" in sensors_dict:
                try:
                    right_cam = sensors_dict["right_wrist_camera"]
                    right_rgb = right_cam.data.output["rgb"][0]
                    if right_rgb.dtype != torch.uint8:
                        right_rgb = (right_rgb * 255).to(torch.uint8)
                    processed_obs["video.right_wrist_view"] = right_rgb.unsqueeze(0).to(device)
                except Exception as e:
                    print(f"⚠️ Failed to get right wrist camera: {e}")
    except Exception as e:
        print(f"⚠️ Failed to access scene sensors: {e}")
        traceback.print_exc()

    # Process robot joint state from observation
    try:
        dex3_states = obs["policy"]["robot_gipper_state"]  # (bs, 14)
        g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (bs, 14) - arms only

        processed_obs["state.left_arm"] = g129_shoulder_states[:, :7].to(device)
        processed_obs["state.right_arm"] = g129_shoulder_states[:, 7:14].to(device)
        processed_obs["state.left_hand"] = dex3_states[:, :7].to(device)
        processed_obs["state.right_hand"] = dex3_states[:, 7:14].to(device)
    except Exception as e:
        print(f"⚠️ Failed to process robot state: {e}")
        traceback.print_exc()
        processed_obs["state.left_arm"] = torch.zeros(1, 7, device=device)
        processed_obs["state.right_arm"] = torch.zeros(1, 7, device=device)
        processed_obs["state.left_hand"] = torch.zeros(1, 7, device=device)
        processed_obs["state.right_hand"] = torch.zeros(1, 7, device=device)

    return processed_obs


def create_grid_video(
    results: List[Dict[str, Any]],
    video_dir: str,
    grid_layout: str = "4x5",
    model_name: str = "model",
) -> List[str]:
    """Create grid video(s) combining all episodes."""
    num_episodes = len(results)
    if num_episodes == 0:
        return []

    def _tighten_grid_layout(num: int, max_rows: int, max_cols: int) -> tuple[int, int]:
        """Pick a tight (rows, cols) layout within (max_rows, max_cols) for num items.

        Preference order:
        1) Minimize empty slots.
        2) Keep aspect ratio close to the requested max_rows/max_cols.
        3) Minimize total area.
        """
        if num <= 0:
            return 0, 0

        target_ratio = max_rows / max_cols if max_cols else 1.0
        best: tuple[int, int] | None = None
        best_key: tuple[int, float, int] | None = None

        for r in range(1, max_rows + 1):
            for c in range(1, max_cols + 1):
                area = r * c
                if area < num:
                    continue
                empty = area - num
                ratio = r / c if c else 1.0
                ratio_penalty = abs(ratio - target_ratio)
                key = (empty, ratio_penalty, area)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (r, c)

        # Fallback: if constraints are too tight, grow cols first (shouldn't happen with defaults).
        if best is None:
            cols = min(max_cols, max(1, int(math.ceil(math.sqrt(num)))))
            rows = int(math.ceil(num / cols))
            return min(rows, max_rows), cols

        return best

    # Auto-split into 5x5 grids if episodes > 20 (25 episodes per grid).
    if num_episodes > 20:
        rows, cols = 5, 5
        episodes_per_grid = 25
        num_grids = (num_episodes + episodes_per_grid - 1) // episodes_per_grid
        print(f"📊 {num_episodes} episodes detected, creating {num_grids} grid videos (5x5 each)...")
    else:
        rows, cols = map(int, grid_layout.lower().split("x"))
        if num_episodes < rows * cols:
            rows, cols = _tighten_grid_layout(num_episodes, rows, cols)
        episodes_per_grid = rows * cols
        num_grids = 1
        print(f"📊 Creating {rows}x{cols} grid video from {num_episodes} episodes...")

    grid_paths: List[str] = []
    for grid_idx in range(num_grids):
        start_ep = grid_idx * episodes_per_grid
        end_ep = min(start_ep + episodes_per_grid, num_episodes)
        print(f"\n  Grid {grid_idx + 1}/{num_grids}: Episodes {start_ep + 1}-{end_ep}")
        grid_path = _create_single_grid_video(
            results[start_ep:end_ep],
            video_dir,
            rows,
            cols,
            model_name,
            grid_idx,
            start_ep,
        )
        if grid_path:
            grid_paths.append(grid_path)

    return grid_paths


def _create_single_grid_video(
    results: List[Dict[str, Any]],
    video_dir: str,
    rows: int,
    cols: int,
    model_name: str,
    grid_idx: int,
    start_ep: int,
) -> Optional[str]:
    """Create a single grid video."""
    total_slots = rows * cols

    all_episode_frames: List[Optional[List[np.ndarray]]] = []
    max_frames = 0
    for i, result in enumerate(results):
        frames = result.get("video_frames") or None
        ep_num = start_ep + i + 1
        if frames:
            max_frames = max(max_frames, len(frames))
            print(f"    Episode {ep_num}: {len(frames)} frames")
        all_episode_frames.append(frames)

    if max_frames == 0:
        print("⚠️ No frames found to create grid video")
        return None

    sample_frame = next((frames[0] for frames in all_episode_frames if frames), None)
    if sample_frame is None:
        return None

    orig_h, orig_w = sample_frame.shape[:2]
    cell_w = 320
    cell_h = int(orig_h * cell_w / orig_w)
    grid_h = cell_h * rows
    grid_w = cell_w * cols
    print(f"    Grid size: {grid_w}x{grid_h} ({cell_w}x{cell_h} per cell)")
    print(f"    Total frames: {max_frames}")

    video_path = Path(video_dir)
    video_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    start_ep_num = start_ep + 1
    end_ep_num = start_ep + len(results)
    if grid_idx > 0:
        grid_video_file = video_path / f"part{grid_idx + 1}_ep{start_ep_num}-{end_ep_num}_{timestamp}_{model_name}.mp4"
    else:
        grid_video_file = video_path / f"{timestamp}_{model_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    video_writer = cv2.VideoWriter(str(grid_video_file), fourcc, fps, (grid_w, grid_h))

    try:
        for frame_idx in range(max_frames):
            grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for ep_idx in range(min(len(all_episode_frames), total_slots)):
                row, col = divmod(ep_idx, cols)
                frames = all_episode_frames[ep_idx]
                result = results[ep_idx]
                ep_num = start_ep + ep_idx + 1

                if frames:
                    actual_frame_idx = frame_idx % len(frames)
                    frame = cv2.resize(frames[actual_frame_idx], (cell_w, cell_h))

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (cell_w, 25), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                    steps = int(result.get("steps", len(frames)))
                    success = bool(result.get("success", False))
                    text = f"Ep{ep_num} {actual_frame_idx}/{steps}"
                    text_color = (0, 255, 0) if success else (0, 0, 255)
                    cv2.putText(frame, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                else:
                    frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        f"Episode {ep_num}: No video",
                        (10, cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (128, 128, 128),
                        1,
                    )

                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                grid_frame[y1:y2, x1:x2] = frame

            video_writer.write(grid_frame)
            if (frame_idx + 1) % 100 == 0:
                print(f"    Progress: {frame_idx + 1}/{max_frames} frames")
    finally:
        video_writer.release()

    print(f"    ✅ Saved: {grid_video_file.name}")
    return str(grid_video_file)


def check_success(env) -> bool:
    """Check if the task is successful."""
    if hasattr(env, "_task_stage"):
        stage = env._task_stage[0]
        return stage.item() >= 4
    return False


def _apply_cfg_default_pose(env, settle_steps: int = 8) -> None:
    """Force robot to its configured default joint pose (from cfg) after env.reset().

    This is useful when actuator targets or external controllers can pull the robot away
    from the init pose immediately after reset.
    """
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)

    # These defaults are populated from the asset init_state / cfg.
    q = robot.data.default_joint_pos[env_ids].clone()
    qd = robot.data.default_joint_vel[env_ids].clone()

    robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    try:
        robot.set_joint_position_target(q[0] if env.num_envs == 1 else q)
        try:
            robot.set_joint_velocity_target(qd[0] if env.num_envs == 1 else qd)
        except Exception:
            pass
        env.scene.write_data_to_sim()
    except Exception:
        pass

    for _ in range(max(1, int(settle_steps))):
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()


def evaluate_episode(
    env,
    policy,
    *,
    max_steps: int,
    episode_num: int,
    save_video: bool = False,
    action_chunk_size: int = 1,
    frequency_hz: float = 0.0,
    task_description: str = "install trocar from box",
) -> Dict[str, Any]:
    """Evaluate one episode."""
    print(f"\n{'=' * 60}")
    print(f"Episode {episode_num + 1}")
    print(f"{'=' * 60}")

    obs, _ = env.reset()
    try:
        _apply_cfg_default_pose(env, settle_steps=10)
        if hasattr(env, "get_observations"):
            obs = env.get_observations()
        elif hasattr(env, "_get_observations"):
            obs = env._get_observations()
    except Exception as e:
        print(f"⚠️ Failed to apply default pose after reset: {e}")

    action_chunk_size = int(max(1, min(16, action_chunk_size)))
    video_frames: Optional[List[np.ndarray]] = [] if save_video else None
    action_buffer: List[np.ndarray] = []

    total_reward = 0.0
    success = False

    hz = float(frequency_hz or 0.0)
    period_s = 1.0 / hz if hz > 0 else 0.0

    for step in range(max_steps):
        loop_start_time = time.perf_counter()

        if not action_buffer:
            try:
                processed_obs = process_observation(obs, env, device=policy.device)
            except Exception as e:
                print(f"⚠️ Failed to process observation: {e}")
                traceback.print_exc()
                processed_obs = {
                    "state.left_arm": torch.zeros(1, 7, device=policy.device),
                    "state.right_arm": torch.zeros(1, 7, device=policy.device),
                    "state.left_hand": torch.zeros(1, 7, device=policy.device),
                    "state.right_hand": torch.zeros(1, 7, device=policy.device),
                }

            with torch.no_grad():
                try:
                    processed_obs["annotation.human.task_description"] = [task_description]

                    processed_obs_cpu = {
                        k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in processed_obs.items()
                    }
                    action_dict = policy.get_action(processed_obs_cpu)

                    action_chunk = np.concatenate(
                        [np.atleast_1d(action_dict[key]) for key in action_dict.keys()],
                        axis=1,
                    )
                    if action_chunk.shape[1] == 28:
                        action_chunk = np.concatenate([np.zeros((16, 15)), action_chunk], axis=1)  # (16, 43)

                    action_buffer = list(action_chunk[:action_chunk_size])
                except Exception as e:
                    print(f"⚠️ Policy prediction failed at step {step}: {e}")
                    traceback.print_exc()
                    action_buffer = [np.zeros(43)]

        action = action_buffer.pop(0)

        uenv = env.unwrapped
        action_tensor = (
            torch.as_tensor(action, device=uenv.device, dtype=torch.float32).unsqueeze(0).repeat(uenv.num_envs, 1)
        )
        obs, reward, terminated, truncated, _info = env.step(action_tensor)
        total_reward += float(reward.item())

        if period_s > 0:
            elapsed = time.perf_counter() - loop_start_time
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        current_stage = env._task_stage[0].item() if hasattr(env, "_task_stage") else -1

        if video_frames is not None:
            try:
                sensors = getattr(env.scene, "sensors", {}) or {}
                cam = sensors.get("front_camera")
                if cam is not None:
                    frame = cam.data.output["rgb"][0].detach().cpu().numpy()
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    video_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                if step == 0:
                    print(f"⚠️ Failed to capture video frame: {e}")

        if check_success(env):
            success = True
            print(f"✅ Task completed at step {step}! (Stage: {current_stage})")
            break

        if terminated.any() or truncated.any():
            print(f"Episode terminated at step {step} (Stage: {current_stage})")
            break

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{max_steps}, Stage: {current_stage}, Reward: {total_reward:.2f}")

    final_stage = env._task_stage[0].item() if hasattr(env, "_task_stage") else -1
    result: Dict[str, Any] = {
        "success": success,
        "steps": step + 1,
        "total_reward": total_reward,
        "episode": episode_num,
        "final_stage": final_stage,
        "video_frames": video_frames if save_video else None,
    }

    print(f"\nEpisode {episode_num + 1} Results:")
    print(f"  Success: {'✅' if success else '❌'}")
    print(f"  Steps: {result['steps']}")
    print(f"  Total Reward: {total_reward:.2f}")

    return result


def _patch_numpydantic_for_isaacsim() -> None:
    """Patch numpydantic NDArray schema generation for Isaac Sim bundled pydantic.

    Isaac Sim loads its own pydantic/pydantic_core from `omni.kit.pip_archive`.
    With some combinations, `numpydantic`'s NDArray schema generation fails with:
      MissingDefinitionError: any-shape-array-... -> InvalidSchemaError

    We cannot monkeypatch `NDArray.__get_pydantic_core_schema__` directly because
    NDArray is implemented with nptyping's metaclass which forbids setting attrs
    (see errors like "Cannot set values to nptyping.NDArray.").

    For evaluation/inference we don't need detailed NDArray JSON schema. So we
    patch `numpydantic.ndarray.make_json_schema` to return a simple schema that
    avoids $defs / definition refs, preventing pydantic's schema cleaner from
    crashing.
    """
    import numpydantic.ndarray as _np_ndarray  # type: ignore
    import numpydantic.schema as _np_schema  # type: ignore

    def _safe_make_json_schema(shape, dtype, _handler):  # noqa: ANN001,ARG001
        # Minimal, ref-free schema. Enough for pydantic metadata without triggering
        # pydantic's internal `$defs` resolution/cleaning.
        return {
            "type": "array",
            "items": {},
            "shape": str(shape),
            "dtype": str(dtype),
        }

    # Patch both the module-level binding used inside NDArray.__get_pydantic_core_schema__
    # and the original function in numpydantic.schema (for completeness).
    _np_ndarray.make_json_schema = _safe_make_json_schema  # type: ignore[attr-defined]
    _np_schema.make_json_schema = _safe_make_json_schema  # type: ignore[attr-defined]


_patch_numpydantic_for_isaacsim()


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Gr00t Policy Direct Evaluation in IsaacLab")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Model: {args_cli.model_path or '<test>'}")
    print("=" * 60)

    test_mode = bool(getattr(args_cli, "test", False))
    if test_mode:
        print("✅ Test mode enabled (dummy policy, no checkpoint needed)")
    elif not args_cli.model_path:
        print("❌ --model_path is required unless --test is set")
        return

    # Parse environment configuration
    print("\n[1/4] Loading environment configuration...")
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        print("✅ Environment config loaded")
    except Exception as e:
        print(f"❌ Failed to parse environment configuration: {e}")
        return

    # Create environment
    print("\n[2/4] Creating environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    set_viewport_camera("/World/envs/env_0/Robot/d435_link/front_cam")

    # Load Gr00t policy
    print("\n[3/4] Loading Gr00t policy...")
    try:
        if test_mode:
            import numpy as _np

            class _DummyPolicy:
                def __init__(self, device: str):
                    self.device = device

                def get_action(self, _obs):  # noqa: ANN001
                    # One chunk of 16 actions; each action is a 43-D vector.
                    return {"actions": _np.zeros((16, 43), dtype=_np.float32)}

            policy = _DummyPolicy(args_cli.device)
            print("✅ Dummy policy ready")
        else:
            import sys
            sys.path.append("/home/nvidia/workspace/yunl/unitree_IL_lerobot")
            from gr00t.model.policy import Gr00tPolicy
            from gr00t.experiment.data_config import UnitreeG1SimDataConfig

            data_config = UnitreeG1SimDataConfig()
            modality_config = data_config.modality_config()
            modality_transform = data_config.transform()

            policy = Gr00tPolicy(
                model_path=args_cli.model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag="new_embodiment",
                device=args_cli.device,
            )
            print("✅ Gr00t policy loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        traceback.print_exc()
        return

    # Run evaluation
    print("\n[4/4] Running evaluation...")
    print("=" * 60)

    results = []

    # Extract model name from model path
    model_name = "test" if test_mode else Path(args_cli.model_path).stem  # Get filename without extension

    try:
        for episode in range(args_cli.num_episodes):
            result = evaluate_episode(
                env,
                policy,
                max_steps=args_cli.max_steps,
                episode_num=episode,
                save_video=args_cli.save_video,
                action_chunk_size=args_cli.action_chunk_size,
                frequency_hz=float(getattr(args_cli, "frequency", 0.0) or 0.0),
                task_description=args_cli.task_description,
            )
            results.append(result)

            if episode + 1 < args_cli.num_episodes:
                time.sleep(1.0)

        # Calculate statistics
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        successes = sum(1 for r in results if r["success"])
        success_rate = successes / len(results) * 100

        success_steps = [r["steps"] for r in results if r["success"]]
        avg_success_steps = np.mean(success_steps) if success_steps else 0

        print("\n📊 Overall Statistics:")
        print(f"  Total Episodes: {len(results)}")
        print(f"  Success Number: {successes}")
        print(f"  Success Rate: {success_rate:.1f}%")
        if success_steps:
            print(f"  Average Steps (Success): {avg_success_steps:.1f}")

        print("\n📋 Episode-by-Episode Results:")
        for r in results:
            status = "✅" if r["success"] else "❌"
            print(f"  Ep {r['episode']+1:2d}: {status} | Steps: {r['steps']:4d} | Reward: {r['total_reward']:7.2f}")

        # Save results to file
        results_file = Path("./eval_results") / f"results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w") as f:
            f.write("Evaluation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Task: {args_cli.task}\n")
            f.write(f"Model: {args_cli.model_path}\n")
            f.write(f"Episodes: {args_cli.num_episodes}\n")
            f.write(f"Action Chunk Size: {args_cli.action_chunk_size}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write(f"Average Success Steps: {avg_success_steps:.1f}\n")
            f.write("\nDetailed Results:\n")
            for r in results:
                status = "SUCCESS" if r["success"] else "FAILED"
                stage = r.get("final_stage", -1)
                f.write(
                    f"  Episode {r['episode'] + 1}: {status}"
                    f" | Stage: {stage}/4"
                    f" | Steps: {r['steps']}"
                    f" | Reward: {r['total_reward']:.3f}\n"
                )

        print(f"\n💾 Results saved to: {results_file}")

        # Create grid video combining all episodes
        if args_cli.save_video:
            try:
                print(f"\n{'='*60}")
                print("Creating Grid Video(s)")
                print(f"{'='*60}")

                grid_video_paths = create_grid_video(results, args_cli.video_dir, args_cli.grid_layout, model_name)
                if grid_video_paths:
                    print(f"\n🎬 {len(grid_video_paths)} grid video(s) created successfully!")
                    for i, path in enumerate(grid_video_paths):
                        print(f"  Grid {i+1}: {Path(path).name}")
            except Exception as e:
                print(f"\n❌ Failed to create grid video: {e}")
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n✅ Cleaning up", flush=True)
        try:
            env.close()
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close environment: {e}", flush=True)


if __name__ == "__main__":
    main()
