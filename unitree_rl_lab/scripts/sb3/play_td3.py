# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an Isaac Lab task with a Stable-Baselines3 TD3 checkpoint."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play an Isaac Lab task with Stable-Baselines3 TD3.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help="Root directory for SB3 logs. Defaults to logs/sb3 inside the repo.",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Experiment folder name. Defaults to task name in lowercase with dashes replaced.",
)
parser.add_argument(
    "--draw_lidar_lines",
    action="store_true",
    default=False,
    help="Draw debug line segments for lidar rays during playback when a lidar sensor is available.",
)
parser.add_argument(
    "--draw_lidar_envs",
    type=int,
    default=8,
    help="Number of environments to visualize lidar lines for when --draw_lidar_lines is enabled.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from stable_baselines3 import TD3

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

try:
    from omni.debugdraw import get_debug_draw_interface as _get_debug_draw_interface
except ImportError:
    _get_debug_draw_interface = None

_LIDAR_LINE_COLOR = 0xFF00FFFF


def _find_latest_checkpoint(log_root_path: str) -> str:
    checkpoint_paths = sorted(Path(log_root_path).glob("**/*.zip"), key=lambda path: path.stat().st_mtime)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No TD3 checkpoints found under: {log_root_path}")
    return str(checkpoint_paths[-1])


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    experiment_name = args_cli.experiment_name or args_cli.task.lower().replace("-", "_")
    if args_cli.log_root is not None:
        log_root_path = os.path.abspath(os.path.join(args_cli.log_root, "sb3", experiment_name))
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", experiment_name))

    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint_path = _find_latest_checkpoint(log_root_path)

    print(f"[INFO] Loading TD3 checkpoint from: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(checkpoint_path), "videos", "play_td3"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env)

    should_draw_lidar_lines = args_cli.draw_lidar_lines and not args_cli.headless and _get_debug_draw_interface is not None
    debug_draw = _get_debug_draw_interface() if should_draw_lidar_lines else None

    model = TD3.load(checkpoint_path, env=env, device=args_cli.device if args_cli.device is not None else "auto")

    obs = env.reset()
    timestep = 0
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(actions)

        if debug_draw is not None:
            lidar_sensor = env.unwrapped.scene.sensors.get("lidar")
            if lidar_sensor is not None:
                num_envs_to_draw = min(args_cli.draw_lidar_envs, lidar_sensor.data.pos_w.shape[0])
                for env_id in range(num_envs_to_draw):
                    ray_hits = lidar_sensor.data.ray_hits_w[env_id]
                    sensor_origin = lidar_sensor.data.pos_w[env_id]
                    num_rays = ray_hits.shape[0]
                    stride = max(1, num_rays // 64)
                    for hit in ray_hits[::stride]:
                        start = tuple(sensor_origin.tolist())
                        end = tuple(hit.tolist())
                        debug_draw.draw_line(start, _LIDAR_LINE_COLOR, 2.0, end, _LIDAR_LINE_COLOR, 2.0)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
