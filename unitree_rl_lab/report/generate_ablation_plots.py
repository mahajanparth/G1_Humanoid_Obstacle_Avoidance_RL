"""
Generate NeurIPS report plots from pre-extracted CSV ablation files.

Data  : /media/parth/Elements/parth/trained_checkpoints/ablation_logs-td3-ppo-no-lidar/
Output: report/TD3-PPO-no-lidar-empty-world/
        report/TD3-PPO-no-lidar-obstacle-world/
"""

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Publication-style rcParams ─────────────────────────────────────────────────
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

PPO_COLOR = "#2271B5"   # deep blue
TD3_COLOR = "#D64E12"   # burnt orange

DATA_ROOT  = "/media/parth/Elements/parth/trained_checkpoints/ablation_logs-td3-ppo-no-lidar"
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ────────────────────────────────────────────────────────────────────

def load(world: str, algo: str, metric: str) -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, world, "csv", f"{algo}_{metric}.csv")
    return pd.read_csv(path)


def plot_pair(world: str, metric: str, title: str, ylabel: str,
              out_dir: str, out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for algo, color, label in [
        ("ppo", PPO_COLOR, "PPO"),
        ("td3", TD3_COLOR, "TD3"),
    ]:
        df   = load(world, algo, metric)
        x    = df["step"].values.astype(float)
        x    = x / x[-1]   # normalise to [0, 1]
        mean = df["rolling_mean"].values
        std  = df["rolling_std"].values

        ax.plot(x, mean, color=color, label=label, zorder=3)
        ax.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.15, zorder=2)

    ax.set_title(title, pad=7)
    ax.set_xlabel("Training Progress (fraction of total)")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9, edgecolor="0.8")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_file)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [OK] {os.path.relpath(out_path, REPORT_DIR)}")


# ── Empty-world plots ──────────────────────────────────────────────────────────
print("\n── Empty-world (no LiDAR) ──────────────────────────────────────────────")

EMPTY_DATA = "TD3-PPO-no-lidar-empty-world"
EMPTY_OUT  = os.path.join(REPORT_DIR, "TD3-PPO-no-lidar-empty-world")

EMPTY_PLOTS = [
    ("mean_episode_length",    "Mean Episode Length",       "Steps",
     "mean_episode_length_td3_vs_ppo_empty_world_no_lidar_ablation.png"),
    ("timeout_termination",    "Timeout Termination Rate",  "Rate",
     "timeout_td3_vs_ppo_empty_world_no_lidar_ablation.png"),
    ("base_height_surrogate",  "Base-Height Surrogate",     "Value",
     "base_height_surrogate_td3_vs_ppo_empty_world_no_lidar_ablation.png"),
    ("bad_orientation_surrogate", "Bad-Orientation Surrogate", "Value",
     "bad_orientation_surrogate_td3_vs_ppo_empty_world_no_lidar_ablation.png"),
    ("velocity_tracking_reward", "Velocity Tracking Reward", "Reward",
     "velocity_tracking_reward_td3_vs_ppo_empty_world_no_lidar_ablation.png"),
]

for metric, title, ylabel, fname in EMPTY_PLOTS:
    plot_pair(EMPTY_DATA, metric, title, ylabel, EMPTY_OUT, fname)


# ── Obstacle-world plots ───────────────────────────────────────────────────────
print("\n── Obstacle-world (no LiDAR) ───────────────────────────────────────────")

OBS_DATA = "TD3-PPO-no-lidar-obstacle-world"
OBS_OUT  = os.path.join(REPORT_DIR, "TD3-PPO-no-lidar-obstacle-world")

OBS_PLOTS = [
    ("mean_episode_length",       "Mean Episode Length",          "Steps",
     "mean_episode_length_td3_vs_ppo_obstacle_world_no_lidar_ablation.png"),
    ("timeout_termination",       "Timeout Termination Rate",     "Rate",
     "timeout_td3_vs_ppo_obstacle_world_no_lidar_ablation.png"),
    ("base_height_termination",   "Base-Height Termination Rate", "Rate",
     "base_height_termination_td3_vs_ppo_obstacle_world_no_lidar_ablation.png"),
    ("bad_orientation_termination", "Bad-Orientation Termination", "Rate",
     "bad_orientation_td3_vs_ppo_obstacle_world_no_lidar_ablation.png"),
    ("velocity_tracking_reward",  "Velocity Tracking Reward",     "Reward",
     "velocity_tracking_reward_td3_vs_ppo_obstacle_world_no_lidar_ablation.png"),
    ("object_contact_reward",     "Object-Contact Reward",        "Reward",
     "object_hit_td3_vs_ppo_obstacle_world_no_lidar_ablation.png"),
]

for metric, title, ylabel, fname in OBS_PLOTS:
    plot_pair(OBS_DATA, metric, title, ylabel, OBS_OUT, fname)

print("\nAll plots saved.\n")
