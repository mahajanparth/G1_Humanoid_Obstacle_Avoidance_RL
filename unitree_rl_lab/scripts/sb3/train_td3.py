# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an Isaac Lab task with Stable-Baselines3 TD3."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3 TD3.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="TD3 initial learning rate.")
parser.add_argument(
    "--lr_schedule",
    type=str,
    default="constant",
    choices=["constant", "linear", "cosine"],
    help="Learning rate schedule: constant, linear decay, or cosine annealing.",
)
parser.add_argument("--lr_final", type=float, default=1e-6, help="Final learning rate for linear/cosine schedules.")
parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Replay buffer size.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--learning_starts", type=int, default=10_000, help="Number of steps before updates start.")
parser.add_argument("--train_freq", type=int, default=1, help="Training frequency in environment steps.")
parser.add_argument("--gradient_steps", type=int, default=4, help="Gradient steps per update.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (element-wise).")
parser.add_argument("--policy_delay", type=int, default=8, help="Policy delay for TD3.")
parser.add_argument("--actor_lr_scale", type=float, default=0.1, help="Actor LR = learning_rate * actor_lr_scale (critic learns faster).")
parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--save_freq", type=int, default=50_000, help="Checkpoint save frequency.")
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help="Root directory for training logs. Defaults to logs/sb3 inside the repo.",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Experiment folder name. Defaults to task name in lowercase with dashes replaced.",
)
parser.add_argument("--run_name", type=str, default=None, help="Optional run name suffix.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import unitree_rl_lab.tasks  # noqa: F401
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class TD3NearZeroWarmup(TD3):
    """TD3 that uses the actor (not random actions) during the warmup phase.

    SB3's default TD3 samples completely random actions before `learning_starts`,
    which causes a humanoid to fall immediately every episode. Since we initialise
    the actor output layer to near-zero weights, using the actor from step 0 keeps
    the robot upright and fills the replay buffer with quality transitions.
    """

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        # Always use the actor — skip SB3's random-action warmup phase entirely.
        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        if action_noise is not None:
            unscaled_action = np.clip(unscaled_action + action_noise(), -1.0, 1.0)
        scaled_action = self.policy.scale_action(unscaled_action)
        buffer_action = self.policy.unscale_action(scaled_action)
        return scaled_action, buffer_action


def make_lr_schedule(schedule: str, initial_lr: float, final_lr: float):
    """Return a learning rate callable compatible with SB3 (progress_remaining: 1→0)."""
    if schedule == "linear":
        def linear(progress_remaining: float) -> float:
            return final_lr + progress_remaining * (initial_lr - final_lr)
        return linear
    elif schedule == "cosine":
        def cosine(progress_remaining: float) -> float:
            return final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return cosine
    else:
        return initial_lr


# Maps SB3's built-in tag names → RSL-RL PPO tag names for 1-to-1 TensorBoard comparison.
_SB3_TO_PPO = {
    "rollout/ep_rew_mean":  "Train/mean_reward",
    "rollout/ep_len_mean":  "Train/mean_episode_length",
    "train/actor_loss":     "Loss/actor",
    "train/critic_loss":    "Loss/value",
    "train/learning_rate":  "Loss/learning_rate",
    "time/fps":             "Perf/total_fps",
    "time/total_timesteps": "Train/total_timesteps",
}


class EpisodeInfoTensorboardCallback(BaseCallback):
    """Log all Isaac Lab data with the same TensorBoard tag names as RSL-RL PPO.

    Three sources are unified:
    1. info["episode"] — Isaac Lab extras returned at episode end. These already
       carry full-path keys (Episode_Reward/..., Episode_Termination/...,
       Metrics/..., Curriculum/...) matching RSL-RL exactly. We log them as-is.
    2. SB3 built-in tags — remapped via _SB3_TO_PPO so rollout/ep_rew_mean
       becomes Train/mean_reward, train/critic_loss becomes Loss/value, etc.
    3. Per-step reward terms — accumulated from _step_reward and logged each
       step so the curves are smooth (same signal RSL-RL logs per iteration).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._isaac_env = None

    def _unwrap_env(self):
        if self._isaac_env is not None:
            return self._isaac_env
        env = self.training_env
        while hasattr(env, "env"):
            env = env.env
        while hasattr(env, "unwrapped") and env.unwrapped is not env:
            env = env.unwrapped
        self._isaac_env = env
        return env

    def _on_step(self) -> bool:
        # ── 1. Episode-boundary data from Isaac Lab (full-path keys) ──────────
        #    info["episode"] keys ARE already "Episode_Reward/term",
        #    "Episode_Termination/time_out", "Metrics/...", "Curriculum/..." —
        #    log them verbatim so they match RSL-RL tags 1-to-1.
        infos = self.locals.get("infos", [])
        episode_data: dict[str, list[float]] = {}
        for info in infos:
            ep = info.get("episode")
            if not ep:
                continue
            for key, value in ep.items():
                if key in {"r", "l", "t"}:
                    continue
                if isinstance(value, (int, float)):
                    episode_data.setdefault(key, []).append(float(value))

        for key, values in episode_data.items():
            self.logger.record(key, float(sum(values) / len(values)))

        # ── 2. Per-step reward terms from _step_reward [num_envs, num_terms] ──
        #    Gives a smooth per-step curve in addition to the episodic values.
        env = self._unwrap_env()
        rm = getattr(env, "reward_manager", None)
        if rm is not None and hasattr(rm, "_step_reward") and hasattr(rm, "_term_names"):
            step_rew = rm._step_reward
            for idx, name in enumerate(rm._term_names):
                self.logger.record(
                    f"Step_Reward/{name}",
                    float(step_rew[:, idx].mean().item()),
                )

        # ── 3. Remap SB3 built-in tags → PPO names ────────────────────────────
        #    name_to_value is the pending dict before the next logger.dump().
        #    We read whatever SB3 already recorded this step and add PPO copies.
        ntv = getattr(self.logger, "name_to_value", {})
        for sb3_key, ppo_key in _SB3_TO_PPO.items():
            if sb3_key in ntv:
                self.logger.record(ppo_key, ntv[sb3_key])

        return True


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    experiment_name = args_cli.experiment_name or args_cli.task.lower().replace("-", "_")
    if args_cli.log_root is not None:
        log_root_path = os.path.abspath(os.path.join(args_cli.log_root, "sb3", experiment_name))
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        run_name += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = Sb3VecEnvWrapper(env, fast_variant=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args_cli.save_freq // env.num_envs),
        save_path=log_dir,
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    lr = make_lr_schedule(args_cli.lr_schedule, args_cli.learning_rate, args_cli.lr_final)

    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        buffer_size=args_cli.buffer_size,
        learning_starts=args_cli.learning_starts,
        batch_size=args_cli.batch_size,
        tau=args_cli.tau,
        gamma=args_cli.gamma,
        train_freq=(args_cli.train_freq, "step"),
        gradient_steps=args_cli.gradient_steps,
        policy_delay=args_cli.policy_delay,
        tensorboard_log=log_root_path,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=dict(net_arch=[1024, 1024, 512, 256, 128]),
        seed=args_cli.seed,
    )

    if args_cli.checkpoint:
        model = TD3NearZeroWarmup.load(args_cli.checkpoint, env=env, device=model_kwargs["device"])
    else:
        model = TD3NearZeroWarmup(**model_kwargs)
        # Initialize actor output layer to near-zero so the robot starts with minimal
        # actions (same behaviour as PPO's small last-layer init). Combined with
        # TD3NearZeroWarmup, the actor (not random actions) is used from step 0,
        # keeping the robot upright and filling the buffer with quality transitions.
        import torch.nn as nn
        for actor in [model.actor, model.actor_target]:
            last_layer = actor.mu[-2]  # mu is Sequential ending in Tanh; [-2] is the final Linear
            nn.init.uniform_(last_layer.weight, -1e-3, 1e-3)
            nn.init.constant_(last_layer.bias, 0.0)

    # Actor LR scaling — critic should converge faster than actor.
    # Lowering actor LR prevents it from chasing a poorly-calibrated critic.
    actor_lr = args_cli.learning_rate * args_cli.actor_lr_scale
    for param_group in model.actor.optimizer.param_groups:
        param_group["lr"] = actor_lr

    # Gradient norm clipping — wraps each optimizer's step() so that
    # clip_grad_norm_ fires before the weight update. This is more principled
    # than element-wise clamping and directly prevents actor/critic divergence.
    def _wrap_with_grad_clip(optimizer, params, max_norm: float):
        _original_step = optimizer.step
        def _clipped_step(*args, **kwargs):
            torch.nn.utils.clip_grad_norm_(list(params), max_norm)
            return _original_step(*args, **kwargs)
        optimizer.step = _clipped_step

    _wrap_with_grad_clip(model.actor.optimizer, model.actor.parameters(), args_cli.max_grad_norm)
    _wrap_with_grad_clip(model.critic.optimizer, model.critic.parameters(), args_cli.max_grad_norm)

    callback = CallbackList([checkpoint_callback, EpisodeInfoTensorboardCallback()])
    model.learn(total_timesteps=args_cli.total_timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(log_dir, "checkpoint"))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
