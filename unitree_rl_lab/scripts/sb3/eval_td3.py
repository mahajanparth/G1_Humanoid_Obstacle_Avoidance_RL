# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained SB3 TD3 policy for a fixed number of episodes and report stats."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an SB3 TD3 policy.")
parser.add_argument("--task",         type=str,  required=True,  help="Task to evaluate on.")
parser.add_argument("--num_envs",     type=int,  default=64,     help="Parallel environments.")
parser.add_argument("--num_episodes", type=int,  default=500,    help="Total episodes to collect.")
parser.add_argument("--checkpoint",   type=str,  required=True,  help="Path to .zip TD3 checkpoint.")
parser.add_argument("--out",          type=str,  default=None,   help="Optional JSON output path.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import statistics
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import TD3

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading TD3 checkpoint: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = Sb3VecEnvWrapper(env)

    model = TD3.load(checkpoint_path, env=env, device=args_cli.device if args_cli.device else "auto")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    ep_returns     = []
    ep_lengths     = []
    reward_sums    = defaultdict(list)

    episodes_done  = 0
    current_return = torch.zeros(args_cli.num_envs)
    current_length = torch.zeros(args_cli.num_envs)

    obs = env.reset()

    print(f"[INFO] Collecting {args_cli.num_episodes} episodes across {args_cli.num_envs} envs...")

    while episodes_done < args_cli.num_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        # rewards/dones are numpy arrays of shape [num_envs]
        current_return += torch.tensor(rewards, dtype=torch.float32)
        current_length += 1

        for i, done in enumerate(dones):
            if done:
                ep_returns.append(current_return[i].item())
                ep_lengths.append(current_length[i].item())
                current_return[i] = 0.0
                current_length[i] = 0.0
                episodes_done += 1

                # Extract episode-level info from Isaac Lab extras
                # Sb3VecEnvWrapper passes Isaac Lab extras["log"] as part of info
                info = infos[i] if isinstance(infos, (list, tuple)) else {}

                # Isaac Lab episode stats are under "log" key in the extras dict,
                # which SB3 wrapper propagates into info
                ep_log = info.get("log", {})
                # Also check direct keys (some wrappers flatten the log)
                for key, val in ep_log.items():
                    if isinstance(val, (int, float)):
                        reward_sums[key].append(float(val))
                    elif hasattr(val, "item"):
                        reward_sums[key].append(float(val))

                if episodes_done % 50 == 0:
                    print(f"  {episodes_done}/{args_cli.num_episodes} episodes", flush=True)

    env.close()

    # ── Compute statistics ────────────────────────────────────────────────────
    def fmt(vals):
        if not vals:
            return "N/A"
        return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals) if len(vals) > 1 else 0:.4f}"

    print("\n" + "=" * 60)
    print(f"Task      : {args_cli.task}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Episodes  : {episodes_done}")
    print("=" * 60)
    print(f"  Mean Episode Return : {fmt(ep_returns)}")
    print(f"  Mean Episode Length : {fmt(ep_lengths)}")
    for key in sorted(reward_sums.keys()):
        vals = reward_sums[key]
        print(f"  {key:<50s}: {fmt(vals)}")
    print("=" * 60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    stats = {
        "task":        args_cli.task,
        "checkpoint":  checkpoint_path,
        "episodes":    episodes_done,
        "mean_return": statistics.mean(ep_returns),
        "std_return":  statistics.stdev(ep_returns) if len(ep_returns) > 1 else 0,
        "mean_length": statistics.mean(ep_lengths),
        "std_length":  statistics.stdev(ep_lengths) if len(ep_lengths) > 1 else 0,
        "breakdown":   {k: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0}
                        for k, v in reward_sums.items()},
    }

    out_path = args_cli.out or os.path.join(
        os.path.dirname(checkpoint_path),
        f"eval_{args_cli.task.replace('-', '_').replace('/', '_')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[INFO] Stats saved to: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
