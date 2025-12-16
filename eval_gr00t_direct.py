#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Direct evaluation of Gr00t policy in IsaacLab simulation
No DDS, no SDK - direct env observation to policy
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import cv2

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Evaluate Gr00t Policy in IsaacLab")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Surgical-G129-Dex3-Joint-Box", help="task name")
parser.add_argument("--model_path", type=str, required=True, help="path to Gr00t model checkpoint")
parser.add_argument("--num_episodes", type=int, default=10, help="number of evaluation episodes")
parser.add_argument("--max_steps", type=int, default=500, help="max steps per episode")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--save_video", action="store_true", help="save video of evaluation")
parser.add_argument("--video_dir", type=str, default="./eval_videos", help="directory to save videos")
parser.add_argument("--grid_layout", type=str, default="4x5", help="grid layout (e.g., '4x5' for 4 rows and 5 columns)")
parser.add_argument("--action_chunk_size", type=int, default=1, help="number of actions to use from action chunk (1-16, default: 1)")

# Add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after Isaac Sim is initialized
import gymnasium as gym
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import Gr00t model
sys.path.append("/home/nvidia/workspace/yunl/unitree_IL_lerobot")
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import UnitreeG1SimDataConfig


def process_observation(obs: Dict, env, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """
    Process raw observation from IsaacLab env to Gr00t policy format.
    
    Args:
        obs: Raw observation dict from env (contains "policy" group)
        env: Environment instance (for accessing scene cameras directly)
        device: Device to put tensors on
        
    Returns:
        Processed observation dict for Gr00t policy
        Format:
            - "video.room_view": Front camera (uint8, shape: (1, H, W, 3))
            - "video.left_wrist_view": Left wrist camera (uint8, shape: (1, H, W, 3))
            - "video.right_wrist_view": Right wrist camera (uint8, shape: (1, H, W, 3))
            - "state.left_arm": Left arm joints (float, shape: (1, 7))
            - "state.right_arm": Right arm joints (float, shape: (1, 7))
            - "state.left_hand": Left hand joints (float, shape: (1, 7))
            - "state.right_hand": Right hand joints (float, shape: (1, 7))
    """
    processed_obs = {}
    
    # Get camera images directly from scene sensors
    try:
        if hasattr(env.scene, 'sensors'):
            sensors_dict = env.scene.sensors
            
            # Front camera
            if "front_camera" in sensors_dict:
                try:
                    front_cam = sensors_dict["front_camera"]
                    front_rgb = front_cam.data.output["rgb"][0]  # Get first env, shape: (H, W, 3)
                    
                    # Gr00t expects uint8 format for video observations
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
                    
                    # Gr00t expects uint8 format for video observations
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
                    
                    # Gr00t expects uint8 format for video observations
                    if right_rgb.dtype != torch.uint8:
                        right_rgb = (right_rgb * 255).to(torch.uint8)
                    
                    processed_obs["video.right_wrist_view"] = right_rgb.unsqueeze(0).to(device)
                except Exception as e:
                    print(f"⚠️ Failed to get right wrist camera: {e}")
    
    except Exception as e:
        print(f"⚠️ Failed to access scene sensors: {e}")
        import traceback
        traceback.print_exc()
    
    # Process robot joint state from observation
    try:
        # Get joint states from observation
        # robot_joint_state: (bs, 29) - reordered G1 29DOF joints
        # robot_gipper_state: (bs, 14) - DEX3 joints (7 per hand)
        dex3_states = obs["policy"]['robot_gipper_state']  # (bs, 14)
        g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (bs, 14) - arms only
        
        processed_obs["state.left_arm"] = g129_shoulder_states[:, :7].to(device)      # (bs, 7)
        processed_obs["state.right_arm"] = g129_shoulder_states[:, 7:14].to(device)   # (bs, 7)
        processed_obs["state.left_hand"] = dex3_states[:, :7].to(device)              # (bs, 7)
        processed_obs["state.right_hand"] = dex3_states[:, 7:14].to(device)           # (bs, 7)
    except Exception as e:
        print(f"⚠️ Failed to process robot state: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to zeros
        processed_obs["state.left_arm"] = torch.zeros(1, 7, device=device)
        processed_obs["state.right_arm"] = torch.zeros(1, 7, device=device)
        processed_obs["state.left_hand"] = torch.zeros(1, 7, device=device)
        processed_obs["state.right_hand"] = torch.zeros(1, 7, device=device)
    
    return processed_obs


def create_grid_video(results: List[Dict], video_dir: str, grid_layout: str = "4x5", model_name: str = "model") -> List[str]:
    """Create grid video(s) combining all episodes.
    
    If episodes > 20, automatically splits into multiple 5x5 grids (25 episodes each).
    
    Args:
        results: List of episode results with video_frames
        video_dir: Directory to save the grid video
        grid_layout: Grid layout as "rows x cols" (e.g., "4x5")
        model_name: Model name for filename
    
    Returns:
        List of paths to the created grid video(s)
    """
    num_episodes = len(results)
    
    # Auto-split into 5x5 grids if episodes > 20
    if num_episodes > 20:
        rows, cols = 5, 5  # Force 5x5 layout
        episodes_per_grid = 25
        num_grids = (num_episodes + episodes_per_grid - 1) // episodes_per_grid  # Ceiling division
        print(f"📊 {num_episodes} episodes detected, creating {num_grids} grid videos (5x5 each)...")
    else:
        # Use specified layout for ≤20 episodes
        rows, cols = map(int, grid_layout.split('x'))
        episodes_per_grid = rows * cols
        num_grids = 1
        print(f"📊 Creating {rows}x{cols} grid video from {num_episodes} episodes...")
    
    all_grid_paths = []
    
    # Create each grid video
    for grid_idx in range(num_grids):
        start_ep = grid_idx * episodes_per_grid
        end_ep = min(start_ep + episodes_per_grid, num_episodes)
        grid_results = results[start_ep:end_ep]
        
        print(f"\n  Grid {grid_idx + 1}/{num_grids}: Episodes {start_ep + 1}-{end_ep}")
        
        grid_path = _create_single_grid_video(grid_results, video_dir, rows, cols, model_name, grid_idx, start_ep)
        if grid_path:
            all_grid_paths.append(grid_path)
    
    return all_grid_paths


def _create_single_grid_video(results: List[Dict], video_dir: str, rows: int, cols: int, model_name: str, grid_idx: int, start_ep: int) -> str:
    """Create a single grid video.
    
    Args:
        results: List of episode results for this grid
        video_dir: Directory to save the grid video
        rows: Number of rows in grid
        cols: Number of columns in grid
        model_name: Model name for filename
        grid_idx: Grid index (for multi-grid scenarios)
        start_ep: Starting episode number (0-based)
    
    Returns:
        Path to the created grid video
    """
    total_slots = rows * cols
    
    # Collect all episode frames for this grid
    all_episode_frames = []
    max_frames = 0
    
    for i, result in enumerate(results):
        frames = result.get("video_frames")
        ep_num = start_ep + i + 1
        if frames and len(frames) > 0:
            all_episode_frames.append(frames)
            max_frames = max(max_frames, len(frames))
            print(f"    Episode {ep_num}: {len(frames)} frames")
        else:
            all_episode_frames.append(None)
    
    if not any(all_episode_frames):
        print("⚠️ No frames found to create grid video")
        return None
    
    # Get frame dimensions from first available episode
    sample_frame = None
    for frames in all_episode_frames:
        if frames:
            sample_frame = frames[0]
            break
    
    if sample_frame is None:
        return None
    
    orig_h, orig_w = sample_frame.shape[:2]
    
    # Calculate cell size (each episode will be resized to this)
    cell_w = 320  # Fixed width for each cell
    cell_h = int(orig_h * cell_w / orig_w)  # Maintain aspect ratio
    
    # Create grid video
    grid_h = cell_h * rows
    grid_w = cell_w * cols
    
    print(f"    Grid size: {grid_w}x{grid_h} ({cell_w}x{cell_h} per cell)")
    print(f"    Total frames: {max_frames}")
    
    # Prepare video writer
    video_path = Path(video_dir)
    video_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    # Generate filename with part number for multi-grid scenarios
    start_ep_num = start_ep + 1
    end_ep_num = start_ep + len(results)
    
    if grid_idx > 0:
        # Multi-grid: include part number and episode range
        grid_video_file = video_path / f"grid_{rows}x{cols}_part{grid_idx+1}_ep{start_ep_num}-{end_ep_num}_{timestamp}_{model_name}.mp4"
    else:
        # Single grid
        grid_video_file = video_path / f"grid_{rows}x{cols}_{timestamp}_{model_name}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(str(grid_video_file), fourcc, fps, (grid_w, grid_h))
    
    # Generate each frame of the grid video
    for frame_idx in range(max_frames):
        # Create empty grid frame
        grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Fill each cell in the grid
        ep_idx = 0
        for row in range(rows):
            for col in range(cols):
                if ep_idx >= len(all_episode_frames):
                    break
                
                frames = all_episode_frames[ep_idx]
                result = results[ep_idx]
                
                # Get frame for this episode (loop if shorter than max_frames)
                if frames and len(frames) > 0:
                    actual_frame_idx = frame_idx % len(frames)
                    frame = frames[actual_frame_idx].copy()
                    
                    # Resize to cell size
                    frame = cv2.resize(frame, (cell_w, cell_h))
                    
                    # Add episode info overlay (use global episode number)
                    ep_num = start_ep + ep_idx + 1
                    status = "success" if result["success"] else "failed"
                    steps = result["steps"]
                    reward = result["total_reward"]
                    
                    # Add semi-transparent background for text
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (cell_w, 25), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                    
                    # Add text (use colored text for status)
                    text = f"Ep{ep_num} {actual_frame_idx}/{steps}"
                    text_color = (0, 255, 0) if result["success"] else (0, 0, 255)  # Green for success, Red for failed
                    cv2.putText(frame, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                else:
                    # Create black cell with episode info
                    frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    ep_num = start_ep + ep_idx + 1
                    text = f"Episode {ep_num}: No video"
                    cv2.putText(frame, text, (10, cell_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                # Place frame in grid
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                grid_frame[y1:y2, x1:x2] = frame
                
                ep_idx += 1
        
        video_writer.write(grid_frame)
        
        # Print progress
        if (frame_idx + 1) % 100 == 0:
            print(f"    Progress: {frame_idx + 1}/{max_frames} frames")
    
    video_writer.release()
    
    file_size_mb = grid_video_file.stat().st_size / (1024 * 1024)
    print(f"    ✅ Saved: {grid_video_file.name}")
    print(f"    📦 Size: {file_size_mb:.2f} MB")
    
    return str(grid_video_file)


def check_success(env) -> bool:
    """
    Check if the task is successful.
    
    For surgical task: check if task stage reached 3 (all stages completed)
    """
    if hasattr(env, '_task_stage'):
        stage = env._task_stage[0]  # Get first environment's stage
        return stage.item() >= 4
    
    # Fallback: check through termination
    # This assumes you have a task_success termination condition
    return False


def evaluate_episode(env, policy, max_steps: int, episode_num: int, save_video: bool = False, action_chunk_size: int = 1) -> Dict:
    """
    Evaluate one episode.
    
    Args:
        env: Environment instance
        policy: Policy instance
        max_steps: Maximum steps per episode
        episode_num: Episode index
        save_video: Whether to collect video frames (for grid video)
        action_chunk_size: Number of actions to use from action chunk (1-16)
    
    Returns:
        Dict with episode results: success, steps, reward, video_frames
    """
    print(f"\n{'='*60}")
    print(f"Episode {episode_num + 1}")
    print(f"{'='*60}")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Validate action_chunk_size
    action_chunk_size = max(1, min(16, action_chunk_size))  # Clamp to [1, 16]
    
    # Initialize video recording
    video_frames = [] if save_video else None
    
    # Initialize action buffer for action chunking
    action_buffer = []
    
    total_reward = 0.0
    success = False
    
    for step in range(max_steps):
        # Get action from buffer or policy
        if len(action_buffer) == 0:
            # Buffer is empty, need to get new actions from policy
            # Process observation for policy
            try:
                processed_obs = process_observation(obs, env, device=policy.device)
            except Exception as e:
                print(f"⚠️ Failed to process observation: {e}")
                import traceback
                traceback.print_exc()
                # Use dummy observation as fallback
                processed_obs = {
                    "state.left_arm": torch.zeros(1, 7, device=policy.device),
                    "state.right_arm": torch.zeros(1, 7, device=policy.device),
                    "state.left_hand": torch.zeros(1, 7, device=policy.device),
                    "state.right_hand": torch.zeros(1, 7, device=policy.device),
                }
            
            # Get action from policy
            with torch.no_grad():
                try:
                    # Add task description annotation (required by Gr00t)
                    processed_obs["annotation.human.task_description"] = ["install trocar from box"]
                    
                    # Move all tensors to CPU before passing to policy (for numpy conversion)
                    processed_obs_cpu = {}
                    for k, v in processed_obs.items():
                        if isinstance(v, torch.Tensor):
                            processed_obs_cpu[k] = v.cpu()
                        else:
                            processed_obs_cpu[k] = v
                    
                    # Gr00t policy expects get_action() method which returns a dict
                    # Format: {"left_arm": array, "right_arm": array, "left_hand": array, "right_hand": array}
                    action_dict = policy.get_action(processed_obs_cpu)
                    
                    # Concatenate all action components (following utils.py implementation)
                    action_chunk = np.concatenate(
                        [np.atleast_1d(action_dict[key]) for key in action_dict.keys()],
                        axis=1,
                    )
                    
                    # Pad to 43 dimensions (full G1 DOF)
                    # Action shape: (16, 28) where 16 is chunk size, 28 is action dim
                    # G1 full body: 12 (legs) + 3 (waist) + 28 (arms + hands) = 43
                    # Need to pad 15 zeros (legs + waist) on action dimension (axis=1)
                    if action_chunk.shape[1] == 28:
                        # Pad (16, 15) zeros at the front on action dimension
                        action_chunk = np.concatenate([np.zeros((16, 15)), action_chunk], axis=1)  # Shape: (16, 43)
                    
                    # Extract specified number of actions from chunk
                    # GR00T predicts 16 future actions, use the first N based on action_chunk_size
                    if len(action_chunk.shape) == 2 and action_chunk.shape[0] > 1:
                        # Take first action_chunk_size actions and store in buffer
                        num_actions = min(action_chunk_size, action_chunk.shape[0])
                        action_buffer = [action_chunk[i] for i in range(num_actions)]
                    else:
                        # Single action, add to buffer
                        action_buffer = [action_chunk]
                    
                except Exception as e:
                    print(f"⚠️ Policy prediction failed at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use zero action as fallback
                    action_buffer = [np.zeros(43)]  # 43 DOF for full G1 body
        
        # Pop action from buffer
        action = action_buffer.pop(0)
        
        # Step environment
        # Convert to torch tensor and add batch dimension for num_envs
        action = torch.tensor(action, device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.item()
        
        # Get current task stage for debugging and video
        current_stage = env._task_stage[0].item() if hasattr(env, '_task_stage') else -1
        
        # Capture frame for video (front camera only)
        if save_video and video_frames is not None:
            try:
                sensors = env.scene.sensors if hasattr(env.scene, 'sensors') else {}
                
                # Get front camera frame
                if "front_camera" in sensors:
                    cam = sensors["front_camera"]
                    frame = cam.data.output["rgb"][0].cpu().numpy()  # (H, W, 3)
                    # Convert from float [0,1] to uint8 [0,255] if needed
                    if frame.dtype == np.float32:
                        frame = (frame * 255).astype(np.uint8)
                    # OpenCV uses BGR, convert from RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_frames.append(frame)
            except Exception as e:
                if step == 0:  # Only print warning once
                    print(f"⚠️ Failed to capture video frame: {e}")
        
        # Check success
        if check_success(env):
            success = True
            print(f"✅ Task completed at step {step}! (Stage: {current_stage})")
            break
        
        # Check termination
        if terminated.any() or truncated.any():
            print(f"Episode terminated at step {step} (Stage: {current_stage})")
            break
        
        # Print progress every 50 steps (more frequent to see stage changes)
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{max_steps}, Stage: {current_stage}, Reward: {total_reward:.2f}")
    
    result = {
        "success": success,
        "steps": step + 1,
        "total_reward": total_reward,
        "episode": episode_num,
        "video_frames": video_frames if save_video else None,  # Store for grid video
    }
    
    # Note: Individual episode videos are not saved
    # Frames are collected for grid video only
    
    print(f"\nEpisode {episode_num + 1} Results:")
    print(f"  Success: {'✅' if success else '❌'}")
    print(f"  Steps: {result['steps']}")
    print(f"  Total Reward: {total_reward:.2f}")
    
    return result


def main():
    """Main evaluation function"""
    print("="*60)
    print("Gr00t Policy Direct Evaluation in IsaacLab")
    print("="*60)
    print(f"Task: {args_cli.task}")
    print(f"Model: {args_cli.model_path}")
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"Device: {args_cli.device}")
    print(f"Action Chunk Size: {args_cli.action_chunk_size}")
    print("="*60)
    
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
    
    # Load Gr00t policy
    print("\n[3/4] Loading Gr00t policy...")
    try:
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
        import traceback
        traceback.print_exc()
        return
    
    # Run evaluation
    print("\n[4/4] Running evaluation...")
    print("="*60)
    
    results = []
    
    # Extract model name from model path
    model_name = Path(args_cli.model_path).stem  # Get filename without extension
    
    try:
        for episode in range(args_cli.num_episodes):
            result = evaluate_episode(
                env, 
                policy, 
                args_cli.max_steps, 
                episode,
                save_video=args_cli.save_video,
                action_chunk_size=args_cli.action_chunk_size
            )
            results.append(result)
            
            # Add delay between episodes
            time.sleep(1.0)
        
        # Calculate statistics
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        successes = sum(1 for r in results if r["success"])
        success_rate = successes / len(results) * 100
        
        avg_steps = np.mean([r["steps"] for r in results])
        avg_reward = np.mean([r["total_reward"] for r in results])
        
        success_steps = [r["steps"] for r in results if r["success"]]
        success_rewards = [r["total_reward"] for r in results if r["success"]]
        avg_success_steps = np.mean(success_steps) if success_steps else 0
        avg_success_reward = np.mean(success_rewards) if success_rewards else 0
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Episodes: {len(results)}")
        print(f"  Successes: {successes}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Average Reward: {avg_reward:.2f}")
        if success_steps:
            print(f"  Average Steps (Success): {avg_success_steps:.1f}")
        
        print(f"\n📋 Episode-by-Episode Results:")
        for r in results:
            status = "✅" if r["success"] else "❌"
            print(f"  Ep {r['episode']+1:2d}: {status} | Steps: {r['steps']:4d} | Reward: {r['total_reward']:7.2f}")
        
        # Save results to file
        results_file = Path("./eval_results") / f"results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, "w") as f:
            f.write(f"Evaluation Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"Task: {args_cli.task}\n")
            f.write(f"Model: {args_cli.model_path}\n")
            f.write(f"Episodes: {args_cli.num_episodes}\n")
            f.write(f"Action Chunk Size: {args_cli.action_chunk_size}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write(f"Average Steps: {avg_steps:.1f}\n")
            f.write(f"Average Reward: {avg_reward:.2f}\n")
            f.write(f"\nDetailed Results:\n")
            for r in results:
                status = "SUCCESS" if r["success"] else "FAILED"
                stage = r.get("final_stage", -1)
                f.write(f"  Episode {r['episode']+1}: {status} | Stage: {stage}/4 | Steps: {r['steps']} | Reward: {r['total_reward']:.3f}\n")
        
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
                print(f"\n⚠️ Failed to create grid video: {e}")
                import traceback
                traceback.print_exc()
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            env.close()
        except Exception as e:
            print(f"Failed to close environment: {e}")
        
        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation: {e}")
        
        print("✅ Cleanup completed")


if __name__ == "__main__":
    main()

