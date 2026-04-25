"""
Generate with-LiDAR comparison plots from TensorBoard event files.

Runs used:
  PPO empty world  : RL_RSL/2026-04-14_11-45-23
  PPO obstacle     : RL_RSL/2026-04-14_18-44-30_finetune_dense_obstacle
  TD3 empty world  : SB3/TD3_14   (TD3_15 kept as optional second seed)

Output folders (relative to this script):
  TD3-PPO-lidar-empty-world/
  PPO-lidar-obstacle-world/      (TD3 obstacle run not yet available)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    2.0,
})

PPO_COLOR  = "#2271B5"
TD3_COLOR  = "#D64E12"
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT  = "/media/parth/Elements/parth/trained_checkpoints"

RUNS = {
    "ppo_empty":    os.path.join(CKPT_ROOT, "RL_RSL/2026-04-14_11-45-23"),
    "ppo_obstacle": os.path.join(CKPT_ROOT, "RL_RSL/2026-04-14_18-44-30_finetune_dense_obstacle"),
    "td3_empty":    os.path.join(CKPT_ROOT, "SB3/TD3_14"),
}

WINDOW = 30   # rolling window for smoothing

# PPO: 4096 envs × 24 steps/iter = transitions per iteration
PPO_TRANSITIONS_PER_ITER = 4096 * 24   # 98,304


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_ea(run_key: str) -> EventAccumulator:
    run_dir = RUNS[run_key]
    ef = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))[0]
    ea = EventAccumulator(ef)
    ea.Reload()
    return ea


def cumulative_episodes_ppo(ea: EventAccumulator) -> np.ndarray:
    """Cumulative episode count at each PPO iteration, normalised to [0, 1]."""
    events = ea.Scalars("Train/mean_episode_length")
    ep_lens = np.maximum(np.array([e.value for e in events], dtype=float), 1.0)
    episodes_per_iter = PPO_TRANSITIONS_PER_ITER / ep_lens
    cumeps = np.cumsum(episodes_per_iter)
    return cumeps / cumeps[-1]   # normalise → [0, 1]


def cumulative_episodes_td3(ea: EventAccumulator) -> np.ndarray:
    """Cumulative episode count at each TD3 log point, normalised to [0, 1]."""
    events = ea.Scalars("rollout/ep_len_mean")
    steps   = np.array([e.step  for e in events], dtype=float)
    ep_lens = np.maximum(np.array([e.value for e in events], dtype=float), 1.0)
    deltas  = np.diff(steps, prepend=0.0)
    cumeps  = np.cumsum(deltas / ep_lens)
    return cumeps / cumeps[-1]   # normalise → [0, 1]


def get_series(ea: EventAccumulator, tag: str, x_episodes: np.ndarray):
    """Return (x_episodes, rolling_mean, rolling_std) for a scalar tag.

    The tag series is resampled to align with x_episodes via interpolation
    when lengths differ (e.g. TD3 logs some tags at different frequencies).
    """
    events = ea.Scalars(tag)
    raw_x    = np.arange(len(events), dtype=float)
    raw_vals = np.array([e.value for e in events], dtype=float)

    # Align: interpolate raw_vals onto the x_episodes index grid
    target_x = np.linspace(0, len(raw_x) - 1, len(x_episodes))
    vals = np.interp(target_x, raw_x, raw_vals)

    s    = pd.Series(vals)
    mean = s.rolling(WINDOW, min_periods=1, center=True).mean().values
    std  = s.rolling(WINDOW, min_periods=1, center=True).std().fillna(0).values
    return x_episodes, mean, std


def draw(ax, x, mean, std, color, label):
    ax.plot(x, mean, color=color, label=label, zorder=3)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, zorder=2)


def save(fig, out_dir: str, fname: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {os.path.relpath(path, REPORT_DIR)}")


# ── Load accumulators ──────────────────────────────────────────────────────────
print("Loading TensorBoard event files...")
ea_ppo_empty    = load_ea("ppo_empty")
ea_ppo_obstacle = load_ea("ppo_obstacle")
ea_td3_empty    = load_ea("td3_empty")

# Precompute cumulative episode x-axes
x_eps_ppo_empty    = cumulative_episodes_ppo(ea_ppo_empty)
x_eps_ppo_obstacle = cumulative_episodes_ppo(ea_ppo_obstacle)
x_eps_td3_empty    = cumulative_episodes_td3(ea_td3_empty)
print("  Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# A. Empty-world with LiDAR:  PPO  vs  TD3
# ─────────────────────────────────────────────────────────────────────────────
print("── Empty-world with LiDAR (PPO vs TD3) ────────────────────────────────")

OUT_EMPTY = os.path.join(REPORT_DIR, "TD3-PPO-lidar-empty-world")

# PPO x-axis: iteration. TD3 x-axis: timestep.
# We label both as "Training Step" and note the difference in the caption.

EMPTY_SPECS = [
    # (ppo_tag, td3_tag, title, ylabel, fname)
    (
        "Train/mean_episode_length",
        "rollout/ep_len_mean",
        "Mean Episode Length",
        "Steps",
        "mean_episode_length_td3_vs_ppo_empty_world_lidar.png",
    ),
    (
        "Episode_Termination/time_out",
        "Episode_Termination/time_out",
        "Timeout Termination Rate",
        "Rate",
        "timeout_td3_vs_ppo_empty_world_lidar.png",
    ),
    (
        "Episode_Termination/base_height",
        "Episode_Termination/base_height",
        "Base-Height Termination Rate",
        "Rate",
        "base_height_td3_vs_ppo_empty_world_lidar.png",
    ),
    (
        "Episode_Termination/bad_orientation",
        "Episode_Termination/bad_orientation",
        "Bad-Orientation Termination Rate",
        "Rate",
        "bad_orientation_td3_vs_ppo_empty_world_lidar.png",
    ),
    (
        "Episode_Reward/track_lin_vel_xy",
        "Step_Reward/track_lin_vel_xy",
        "Velocity Tracking Reward",
        "Reward",
        "velocity_tracking_reward_td3_vs_ppo_empty_world_lidar.png",
    ),
]

for ppo_tag, td3_tag, title, ylabel, fname in EMPTY_SPECS:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x, m, s = get_series(ea_ppo_empty, ppo_tag, x_eps_ppo_empty)
    draw(ax, x, m, s, PPO_COLOR, "PPO")
    x, m, s = get_series(ea_td3_empty, td3_tag, x_eps_td3_empty)
    draw(ax, x, m, s, TD3_COLOR, "TD3")
    ax.set_title(title, pad=7)
    ax.set_xlabel("Training Progress (fraction of total)")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    save(fig, OUT_EMPTY, fname)


# ─────────────────────────────────────────────────────────────────────────────
# B. Obstacle-world with LiDAR:  PPO  only  (TD3 run pending)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Obstacle-world with LiDAR (PPO only — TD3 run pending) ─────────────")

OUT_OBS = os.path.join(REPORT_DIR, "PPO-lidar-obstacle-world")

OBS_SPECS = [
    ("Train/mean_episode_length",          "Mean Episode Length",          "Steps",
     "mean_episode_length_ppo_obstacle_world_lidar.png"),
    ("Episode_Termination/time_out",       "Timeout Termination Rate",     "Rate",
     "timeout_ppo_obstacle_world_lidar.png"),
    ("Episode_Termination/base_height",    "Base-Height Termination Rate", "Rate",
     "base_height_ppo_obstacle_world_lidar.png"),
    ("Episode_Termination/bad_orientation","Bad-Orientation Termination",  "Rate",
     "bad_orientation_ppo_obstacle_world_lidar.png"),
    ("Episode_Termination/obstacle_collision", "Obstacle Collision Rate",  "Rate",
     "obstacle_collision_ppo_obstacle_world_lidar.png"),
    ("Episode_Reward/track_lin_vel_xy",    "Velocity Tracking Reward",     "Reward",
     "velocity_tracking_reward_ppo_obstacle_world_lidar.png"),
    ("Episode_Reward/object_hit",          "Object-Hit Penalty",           "Reward",
     "object_hit_ppo_obstacle_world_lidar.png"),
    ("Episode_Reward/lidar_proximity",     "LiDAR Proximity Penalty",      "Reward",
     "lidar_proximity_ppo_obstacle_world_lidar.png"),
]

for tag, title, ylabel, fname in OBS_SPECS:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x, m, s = get_series(ea_ppo_obstacle, tag, x_eps_ppo_obstacle)
    draw(ax, x, m, s, PPO_COLOR, "PPO (obstacle + lidar)")
    ax.set_title(title, pad=7)
    ax.set_xlabel("Training Progress (fraction of total)")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    save(fig, OUT_OBS, fname)

# ─────────────────────────────────────────────────────────────────────────────
# C. PPO with LiDAR:  Empty world  vs  Obstacle world
# ─────────────────────────────────────────────────────────────────────────────
print("\n── PPO with LiDAR: Empty world vs Obstacle world ───────────────────────")

EMPTY_COLOR    = "#2271B5"   # blue  — empty world
OBSTACLE_COLOR = "#4CAF50"   # green — obstacle world

OUT_PPO_COMP = os.path.join(REPORT_DIR, "PPO-lidar-empty-vs-obstacle")

# Common tags present in both runs
PPO_COMP_SPECS = [
    ("Train/mean_episode_length",
     "Mean Episode Length", "Steps",
     "mean_episode_length_ppo_lidar_empty_vs_obstacle.png"),
    ("Episode_Termination/time_out",
     "Timeout Termination Rate", "Rate",
     "timeout_ppo_lidar_empty_vs_obstacle.png"),
    ("Episode_Termination/base_height",
     "Base-Height Termination Rate", "Rate",
     "base_height_ppo_lidar_empty_vs_obstacle.png"),
    ("Episode_Termination/bad_orientation",
     "Bad-Orientation Termination", "Rate",
     "bad_orientation_ppo_lidar_empty_vs_obstacle.png"),
    ("Episode_Reward/track_lin_vel_xy",
     "Velocity Tracking Reward", "Reward",
     "velocity_tracking_reward_ppo_lidar_empty_vs_obstacle.png"),
    ("Episode_Reward/flat_orientation_l2",
     "Flat Orientation Penalty", "Reward",
     "flat_orientation_ppo_lidar_empty_vs_obstacle.png"),
    ("Episode_Reward/alive",
     "Alive Reward", "Reward",
     "alive_ppo_lidar_empty_vs_obstacle.png"),
]

for tag, title, ylabel, fname in PPO_COMP_SPECS:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x, m, s = get_series(ea_ppo_empty,    tag, x_eps_ppo_empty)
    draw(ax, x, m, s, EMPTY_COLOR,    "PPO — Empty world")
    x, m, s = get_series(ea_ppo_obstacle, tag, x_eps_ppo_obstacle)
    draw(ax, x, m, s, OBSTACLE_COLOR, "PPO — Obstacle world")
    ax.set_title(title, pad=7)
    ax.set_xlabel("Training Progress (fraction of total)")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    save(fig, OUT_PPO_COMP, fname)

# Obstacle-only tags — only available in the obstacle run, shown solo with annotation
OBS_ONLY_SPECS = [
    ("Episode_Termination/obstacle_collision",
     "Obstacle Collision Rate", "Rate",
     "obstacle_collision_ppo_lidar_obstacle_only.png"),
    ("Episode_Reward/object_hit",
     "Object-Hit Penalty", "Reward",
     "object_hit_ppo_lidar_obstacle_only.png"),
    ("Episode_Reward/lidar_proximity",
     "LiDAR Proximity Penalty", "Reward",
     "lidar_proximity_ppo_lidar_obstacle_only.png"),
]

for tag, title, ylabel, fname in OBS_ONLY_SPECS:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x, m, s = get_series(ea_ppo_obstacle, tag, x_eps_ppo_obstacle)
    draw(ax, x, m, s, OBSTACLE_COLOR, "PPO — Obstacle world")
    ax.set_title(title, pad=7)
    ax.set_xlabel("Training Progress (fraction of total)")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    save(fig, OUT_PPO_COMP, fname)

print("\nAll plots saved.\n")
