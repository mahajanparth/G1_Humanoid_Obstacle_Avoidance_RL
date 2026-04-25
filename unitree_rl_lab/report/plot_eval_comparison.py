"""
Bar charts comparing obstacle-trained vs empty-world-transfer PPO+LiDAR policies
evaluated on the Dense Obstacle world (Ablation V).
"""

import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(REPORT_DIR, "ablation_v_transfer")
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS = {
    "PPO + LiDAR\n(Obstacle World)": "/media/parth/Elements/parth/trained_checkpoints/RL_RSL/2026-04-14_18-44-30_finetune_dense_obstacle/eval_Unitree_G1_29dof_Dense_Obstacle_FullLidar_Eval.json",
    "PPO + LiDAR\n(Empty World)":    "/media/parth/Elements/parth/working_model/ppo/working_policy/eval_Unitree_G1_29dof_Dense_Obstacle_FullLidar_Eval.json",
}

COLORS = ["#2271B5", "#D64E12"]

data = {}
for label, path in RESULTS.items():
    with open(path) as f:
        data[label] = json.load(f)

labels = list(data.keys())
x      = np.arange(len(labels))
width  = 0.5


def bar_plot(means, stds, title, ylabel, fname, hline=None):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                  color=COLORS, alpha=0.85, zorder=3)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)

    # Annotate each bar with its value, placed just above (or below for negative)
    for bar, val, std in zip(bars, means, stds):
        if val >= 0:
            ypos   = bar.get_height() + std
            va     = "bottom"
        else:
            ypos   = bar.get_height() - std
            va     = "top"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    # Expand y-limits so annotations are not clipped
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    ax.set_ylim(ylo - span * 0.05, yhi + span * 0.18)

    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {os.path.relpath(path, REPORT_DIR)}")


# ── 1. Mean Episode Return ────────────────────────────────────────────────────
m = [data[l]["mean_return"] for l in labels]
s = [data[l]["std_return"]  for l in labels]
bar_plot(m, s, "Mean Episode Return", "Return", "mean_return.png", hline=0)

# ── 2. Mean Episode Length ────────────────────────────────────────────────────
m = [data[l]["mean_length"] for l in labels]
s = [data[l]["std_length"]  for l in labels]
bar_plot(m, s, "Mean Episode Length", "Steps", "mean_episode_length.png")

# ── 3. Timeout Rate ───────────────────────────────────────────────────────────
bd = "breakdown"
m = [data[l][bd]["Episode_Termination/time_out"]["mean"] for l in labels]
s = [data[l][bd]["Episode_Termination/time_out"]["std"]  for l in labels]
bar_plot(m, s, "Timeout Rate", "Rate", "timeout_rate.png")

# ── 4. Bad Orientation Termination ────────────────────────────────────────────
m = [data[l][bd]["Episode_Termination/bad_orientation"]["mean"] for l in labels]
s = [data[l][bd]["Episode_Termination/bad_orientation"]["std"]  for l in labels]
bar_plot(m, s, "Bad-Orientation Termination Rate", "Rate", "bad_orientation_rate.png")

# ── 5. Base Height Termination ────────────────────────────────────────────────
m = [data[l][bd]["Episode_Termination/base_height"]["mean"] for l in labels]
s = [data[l][bd]["Episode_Termination/base_height"]["std"]  for l in labels]
bar_plot(m, s, "Base-Height Termination Rate", "Rate", "base_height_rate.png")

# ── 6. Velocity Tracking Reward ───────────────────────────────────────────────
m = [data[l][bd]["Episode_Reward/track_lin_vel_xy"]["mean"] for l in labels]
s = [data[l][bd]["Episode_Reward/track_lin_vel_xy"]["std"]  for l in labels]
bar_plot(m, s, "Velocity Tracking Reward", "Reward", "velocity_tracking.png")

# ── 7. LiDAR Proximity Penalty ────────────────────────────────────────────────
m = [data[l][bd]["Episode_Reward/lidar_proximity"]["mean"] for l in labels]
s = [data[l][bd]["Episode_Reward/lidar_proximity"]["std"]  for l in labels]
bar_plot(m, s, "LiDAR Proximity Penalty", "Reward", "lidar_proximity.png", hline=0)

# ── 8. Obstacle Collision Rate ────────────────────────────────────────────────
m = [data[l][bd]["Episode_Termination/obstacle_collision"]["mean"] for l in labels]
s = [data[l][bd]["Episode_Termination/obstacle_collision"]["std"]  for l in labels]
bar_plot(m, s, "Obstacle Collision Rate", "Rate", "obstacle_collision_rate.png")

print(f"\nAll plots saved to: {os.path.relpath(OUT_DIR, REPORT_DIR)}/")
