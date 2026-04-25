# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy for a fixed number of episodes and report stats."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy.")
parser.add_argument("--task",         type=str,  required=True,  help="Task to evaluate on.")
parser.add_argument("--num_envs",     type=int,  default=64,     help="Parallel environments.")
parser.add_argument("--num_episodes", type=int,  default=500,    help="Total episodes to collect.")
parser.add_argument("--out",          type=str,  default=None,   help="Optional JSON output path.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)  # registers --checkpoint, --load_run, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import statistics
from collections import defaultdict

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    ep_returns     = []
    ep_lengths     = []
    term_counts    = defaultdict(int)
    reward_sums    = defaultdict(list)

    episodes_done  = 0
    current_return = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    current_length = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)

    try:
        obs, _ = env.get_observations()
    except (TypeError, ValueError):
        obs = env.get_observations()

    print(f"[INFO] Collecting {args_cli.num_episodes} episodes across {args_cli.num_envs} envs...")

    while episodes_done < args_cli.num_episodes:
        with torch.inference_mode():
            actions      = policy(obs)
            obs, rew, dones, extras = env.step(actions)

        current_return += rew
        current_length += 1

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            for idx in done_indices:
                ep_returns.append(current_return[idx].item())
                ep_lengths.append(current_length[idx].item())
                current_return[idx] = 0.0
                current_length[idx] = 0.0
                episodes_done += 1

            # Accumulate episode-level info (termination flags, reward terms)
            # Isaac Lab stores episode stats in extras["log"]
            ep_info = extras.get("log", extras.get("episode", {}))
            for key, val in ep_info.items():
                if isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        # scalar tensor — shared across all envs
                        for _ in done_indices:
                            reward_sums[key].append(val.item())
                    else:
                        # [num_envs] tensor — take done envs only
                        for idx in done_indices:
                            reward_sums[key].append(val[idx].item())
                elif isinstance(val, (int, float)):
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
    print(f"Checkpoint: {os.path.basename(resume_path)}")
    print(f"Episodes  : {episodes_done}")
    print("=" * 60)
    print(f"  Mean Episode Return : {fmt(ep_returns)}")
    print(f"  Mean Episode Length : {fmt(ep_lengths)}")

    # Termination and reward breakdown from extras
    for key in sorted(reward_sums.keys()):
        vals = reward_sums[key]
        print(f"  {key:<50s}: {fmt(vals)}")
    print("=" * 60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    stats = {
        "task":          args_cli.task,
        "checkpoint":    resume_path,
        "episodes":      episodes_done,
        "mean_return":   statistics.mean(ep_returns),
        "std_return":    statistics.stdev(ep_returns) if len(ep_returns) > 1 else 0,
        "mean_length":   statistics.mean(ep_lengths),
        "std_length":    statistics.stdev(ep_lengths) if len(ep_lengths) > 1 else 0,
        "breakdown":     {k: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0}
                          for k, v in reward_sums.items()},
    }

    out_path = args_cli.out or os.path.join(
        os.path.dirname(resume_path),
        f"eval_{args_cli.task.replace('-', '_').replace('/', '_')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[INFO] Stats saved to: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
