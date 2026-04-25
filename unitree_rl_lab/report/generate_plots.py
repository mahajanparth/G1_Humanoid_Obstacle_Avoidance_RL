"""
Generate presentation-ready plots from TensorBoard training logs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "legend.framealpha": 0.85,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

OUT_DIR = "/home/parth/Project_Saraswati_RL/unitree_rl_lab/report/plots"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "/media/parth/Elements/parth/rsl_rl/unitree_g1_29dof_empty_lidar_velocity"

# ── Run definitions ───────────────────────────────────────────────────────────
RUNS = {
    "Empty World\n(pre-train)":        f"{BASE}/2026-04-14_11-45-23",
    "Single Obstacle\n(finetune 1)":   f"{BASE}/2026-04-14_14-37-06_finetune_single_obstacle",
    "Single Obstacle\n(finetune 2)":   f"{BASE}/2026-04-14_14-42-37_finetune_single_obstacle",
    "Single Obstacle\n(finetune 3)":   f"{BASE}/2026-04-14_14-50-27_finetune_single_obstacle",
    "Dense Obstacle\n(finetune 1)":    f"{BASE}/2026-04-14_14-54-33_finetune_dense_obstacle",
    "Dense Obstacle\n(finetune 2)":    f"{BASE}/2026-04-14_18-44-30_finetune_dense_obstacle",
    "Single Obstacle\n(finetune 4)":   f"{BASE}/2026-04-14_18-47-51_finetune_single_obstacle",
}

COLORS = {
    "Empty World\n(pre-train)":        "#2196F3",
    "Single Obstacle\n(finetune 1)":   "#FF9800",
    "Single Obstacle\n(finetune 2)":   "#FF5722",
    "Single Obstacle\n(finetune 3)":   "#E91E63",
    "Dense Obstacle\n(finetune 1)":    "#4CAF50",
    "Dense Obstacle\n(finetune 2)":    "#1B5E20",
    "Single Obstacle\n(finetune 4)":   "#9C27B0",
}

# Grouped for cleaner plots
EMPTY_RUNS  = ["Empty World\n(pre-train)"]
SINGLE_RUNS = ["Single Obstacle\n(finetune 1)", "Single Obstacle\n(finetune 2)",
               "Single Obstacle\n(finetune 3)", "Single Obstacle\n(finetune 4)"]
DENSE_RUNS  = ["Dense Obstacle\n(finetune 1)", "Dense Obstacle\n(finetune 2)"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_scalar(run_path: str, tag: str):
    ea = EventAccumulator(run_path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window=15):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def plot_tag(ax, run_name, run_path, tag, label=None, window=15, alpha_raw=0.15):
    steps, vals = load_scalar(run_path, tag)
    if len(steps) == 0:
        return
    color = COLORS[run_name]
    s = smooth(vals, window)
    ax.plot(steps, vals, color=color, alpha=alpha_raw, linewidth=0.8)
    ax.plot(steps, s, color=color, linewidth=2.2, label=label or run_name.replace("\n", " "))


# ════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW: Mean reward + episode length across ALL runs
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, path in RUNS.items():
    for ax, tag in zip(axes, ["Train/mean_reward", "Train/mean_episode_length"]):
        plot_tag(ax, name, path, tag)

axes[0].set_title("Mean Episode Reward — All Runs")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Mean Reward")
axes[0].legend(loc="upper left", ncol=1, fontsize=9)

axes[1].set_title("Mean Episode Length — All Runs")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Steps")
axes[1].legend(loc="upper left", ncol=1, fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_overview_all_runs.png", bbox_inches="tight")
plt.close()
print("Saved 01_overview_all_runs.png")


# ════════════════════════════════════════════════════════════════════════════
# 2. PRE-TRAINING: Empty world detailed reward breakdown
# ════════════════════════════════════════════════════════════════════════════
reward_tags = [
    ("Episode_Reward/track_lin_vel_xy", "Vel XY tracking"),
    ("Episode_Reward/track_ang_vel_z",  "Ang vel tracking"),
    ("Episode_Reward/gait",             "Gait"),
    ("Episode_Reward/feet_clearance",   "Feet clearance"),
    ("Episode_Reward/flat_orientation_l2", "Orientation"),
    ("Episode_Reward/base_height",      "Base height"),
    ("Episode_Reward/alive",            "Alive bonus"),
    ("Episode_Reward/undesired_contacts", "Undesired contacts"),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
run_name = "Empty World\n(pre-train)"
run_path = RUNS[run_name]

for ax, (tag, label) in zip(axes, reward_tags):
    steps, vals = load_scalar(run_path, tag)
    if len(steps) == 0:
        ax.set_visible(False)
        continue
    ax.plot(steps, vals, color=COLORS[run_name], alpha=0.2, linewidth=0.8)
    ax.plot(steps, smooth(vals, 20), color=COLORS[run_name], linewidth=2.2)
    ax.set_title(label)
    ax.set_xlabel("Iteration")

fig.suptitle("Empty World Pre-Training — Reward Components", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_empty_world_reward_components.png", bbox_inches="tight")
plt.close()
print("Saved 02_empty_world_reward_components.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. SINGLE vs DENSE obstacle — mean reward comparison
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name in SINGLE_RUNS:
    plot_tag(axes[0], name, RUNS[name], "Train/mean_reward")
axes[0].set_title("Single Obstacle — Mean Reward")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Mean Reward")
axes[0].legend(fontsize=9)

for name in DENSE_RUNS:
    plot_tag(axes[1], name, RUNS[name], "Train/mean_reward")
axes[1].set_title("Dense Obstacle — Mean Reward")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Mean Reward")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_obstacle_mean_reward.png", bbox_inches="tight")
plt.close()
print("Saved 03_obstacle_mean_reward.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. TERMINATION breakdown — empty vs obstacle runs
# ════════════════════════════════════════════════════════════════════════════
term_tags = [
    ("Episode_Termination/time_out",       "Timeout"),
    ("Episode_Termination/base_height",    "Fall (height)"),
    ("Episode_Termination/bad_orientation","Bad orientation"),
]
term_colors = ["#2196F3", "#F44336", "#FF9800"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

key_runs = {
    "Empty World":        RUNS["Empty World\n(pre-train)"],
    "Single Obstacle":    RUNS["Single Obstacle\n(finetune 3)"],
    "Dense Obstacle":     RUNS["Dense Obstacle\n(finetune 2)"],
}
run_colors = {"Empty World": "#2196F3", "Single Obstacle": "#E91E63", "Dense Obstacle": "#1B5E20"}

for ax, (tag, label), tc in zip(axes, term_tags, term_colors):
    for run_label, run_path in key_runs.items():
        steps, vals = load_scalar(run_path, tag)
        if len(steps) == 0:
            continue
        ax.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                linewidth=2.2, label=run_label)
    ax.set_title(label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of episodes")
    ax.legend(fontsize=9)

fig.suptitle("Termination Rates Across Environments", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_termination_rates.png", bbox_inches="tight")
plt.close()
print("Saved 04_termination_rates.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. VELOCITY TRACKING error
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for run_label, run_path in key_runs.items():
    color = run_colors[run_label]
    for ax, tag, ylabel in zip(
        axes,
        ["Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw"],
        ["XY Velocity Error (m/s)", "Yaw Velocity Error (rad/s)"],
    ):
        steps, vals = load_scalar(run_path, tag)
        if len(steps) == 0:
            continue
        ax.plot(steps, smooth(vals, 20), color=color, linewidth=2.2, label=run_label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Iteration")

axes[0].set_title("XY Velocity Tracking Error")
axes[1].set_title("Yaw Velocity Tracking Error")
axes[0].legend()
axes[1].legend()

fig.suptitle("Command Tracking Metrics", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_velocity_tracking.png", bbox_inches="tight")
plt.close()
print("Saved 05_velocity_tracking.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. OBSTACLE REWARD terms (object_hit) — obstacle runs only
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

obstacle_runs = {**{n: RUNS[n] for n in SINGLE_RUNS}, **{n: RUNS[n] for n in DENSE_RUNS}}

for name, path in obstacle_runs.items():
    steps, vals = load_scalar(path, "Episode_Reward/object_hit")
    if len(steps) == 0:
        continue
    color = COLORS[name]
    label = name.replace("\n", " ")
    axes[0].plot(steps, vals, color=color, alpha=0.2, linewidth=0.8)
    axes[0].plot(steps, smooth(vals, 20), color=color, linewidth=2.2, label=label)

axes[0].set_title("Object Hit Penalty (lower = fewer hits)")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Reward term value")
axes[0].legend(fontsize=9)

# Episode length for obstacle runs
for name, path in obstacle_runs.items():
    steps, vals = load_scalar(path, "Train/mean_episode_length")
    if len(steps) == 0:
        continue
    color = COLORS[name]
    axes[1].plot(steps, vals, color=color, alpha=0.2, linewidth=0.8)
    axes[1].plot(steps, smooth(vals, 20), color=color, linewidth=2.2,
                 label=name.replace("\n", " "))

axes[1].set_title("Mean Episode Length — Obstacle Runs")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Steps")
axes[1].legend(fontsize=9)

fig.suptitle("Obstacle Avoidance Metrics", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_obstacle_avoidance_metrics.png", bbox_inches="tight")
plt.close()
print("Saved 06_obstacle_avoidance_metrics.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. PPO LOSS curves — empty world pre-training
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
run_path = RUNS["Empty World\n(pre-train)"]
loss_tags = [
    ("Loss/value",     "Value Loss",     "#F44336"),
    ("Loss/surrogate", "Surrogate Loss", "#2196F3"),
    ("Loss/entropy",   "Entropy Loss",   "#4CAF50"),
]
for ax, (tag, label, color) in zip(axes, loss_tags):
    steps, vals = load_scalar(run_path, tag)
    if len(steps) == 0:
        continue
    ax.plot(steps, vals, color=color, alpha=0.2, linewidth=0.8)
    ax.plot(steps, smooth(vals, 20), color=color, linewidth=2.2)
    ax.set_title(label)
    ax.set_xlabel("Iteration")

fig.suptitle("PPO Training Loss — Empty World Pre-Training", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_ppo_loss_curves.png", bbox_inches="tight")
plt.close()
print("Saved 07_ppo_loss_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. LEARNING RATE schedule
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Loss/learning_rate")
    if len(steps) == 0:
        continue
    ax.plot(steps, vals, color=run_colors[run_label], linewidth=2.2, label=run_label)
ax.set_title("Adaptive Learning Rate Schedule")
ax.set_xlabel("Iteration")
ax.set_ylabel("Learning Rate")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_learning_rate.png", bbox_inches="tight")
plt.close()
print("Saved 08_learning_rate.png")


# ════════════════════════════════════════════════════════════════════════════
# 9. POLICY STD (exploration) over training
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Policy/mean_std")
    if len(steps) == 0:
        continue
    ax.plot(steps, smooth(vals, 15), color=run_colors[run_label], linewidth=2.2, label=run_label)
ax.set_title("Policy Standard Deviation (Exploration) Over Training")
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Std")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_policy_std.png", bbox_inches="tight")
plt.close()
print("Saved 09_policy_std.png")


# ════════════════════════════════════════════════════════════════════════════
# 10. MEGA DASHBOARD — presentation summary slide
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: mean reward (all), episode length (key runs), vel error XY, vel error yaw
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2])
ax03 = fig.add_subplot(gs[0, 3])

for name, path in RUNS.items():
    plot_tag(ax00, name, path, "Train/mean_reward", window=20)
ax00.set_title("Mean Reward — All Runs")
ax00.set_xlabel("Iteration"); ax00.set_ylabel("Reward")
ax00.legend(fontsize=6, ncol=1)

for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Train/mean_episode_length")
    if len(vals): ax01.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                            linewidth=2.0, label=run_label)
ax01.set_title("Episode Length"); ax01.set_xlabel("Iteration"); ax01.set_ylabel("Steps")
ax01.legend(fontsize=9)

for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Metrics/base_velocity/error_vel_xy")
    if len(vals): ax02.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                            linewidth=2.0, label=run_label)
ax02.set_title("XY Velocity Error"); ax02.set_xlabel("Iteration"); ax02.set_ylabel("m/s")
ax02.legend(fontsize=9)

for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Metrics/base_velocity/error_vel_yaw")
    if len(vals): ax03.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                            linewidth=2.0, label=run_label)
ax03.set_title("Yaw Velocity Error"); ax03.set_xlabel("Iteration"); ax03.set_ylabel("rad/s")
ax03.legend(fontsize=9)

# Row 1: object hit, termination timeout, fall rate, orientation bad
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax12 = fig.add_subplot(gs[1, 2])
ax13 = fig.add_subplot(gs[1, 3])

for name, path in obstacle_runs.items():
    steps, vals = load_scalar(path, "Episode_Reward/object_hit")
    if len(vals): ax10.plot(steps, smooth(vals, 20), color=COLORS[name],
                            linewidth=2.0, label=name.replace("\n"," "))
ax10.set_title("Object Hit Penalty"); ax10.set_xlabel("Iteration"); ax10.legend(fontsize=7)

for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Episode_Termination/time_out")
    if len(vals): ax11.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                            linewidth=2.0, label=run_label)
ax11.set_title("Timeout Rate"); ax11.set_xlabel("Iteration"); ax11.legend(fontsize=9)

for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Episode_Termination/base_height")
    if len(vals): ax12.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                            linewidth=2.0, label=run_label)
ax12.set_title("Fall Rate"); ax12.set_xlabel("Iteration"); ax12.legend(fontsize=9)

for run_label, run_path in key_runs.items():
    steps, vals = load_scalar(run_path, "Episode_Termination/bad_orientation")
    if len(vals): ax13.plot(steps, smooth(vals, 20), color=run_colors[run_label],
                            linewidth=2.0, label=run_label)
ax13.set_title("Bad Orientation Rate"); ax13.set_xlabel("Iteration"); ax13.legend(fontsize=9)

# Row 2: reward components from empty world
comp_tags = [
    ("Episode_Reward/track_lin_vel_xy", "Vel XY Track", "#2196F3"),
    ("Episode_Reward/gait",             "Gait",         "#4CAF50"),
    ("Episode_Reward/feet_clearance",   "Foot Clearance","#FF9800"),
    ("Episode_Reward/flat_orientation_l2","Orientation", "#9C27B0"),
]
for col, (tag, label, color) in enumerate(comp_tags):
    ax = fig.add_subplot(gs[2, col])
    run_path = RUNS["Empty World\n(pre-train)"]
    steps, vals = load_scalar(run_path, tag)
    if len(vals):
        ax.plot(steps, vals, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(steps, smooth(vals, 20), color=color, linewidth=2.2)
    ax.set_title(label + " (Empty World)")
    ax.set_xlabel("Iteration")

fig.suptitle("Training Dashboard — Unitree G1 29-DoF Obstacle-Aware Locomotion",
             fontsize=18, fontweight="bold", y=0.98)
plt.savefig(f"{OUT_DIR}/10_dashboard.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved 10_dashboard.png")

print(f"\nAll plots saved to: {OUT_DIR}")
