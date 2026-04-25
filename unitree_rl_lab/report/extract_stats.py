"""
Extract final-training statistics from PPO TensorBoard runs.

Reports mean ± std over the last WINDOW_FRAC fraction of training for each
key metric. Output is printed as a LaTeX-ready table and saved to stats.txt.

Runs:
  PPO + LiDAR, Empty world   : RL_RSL/2026-04-14_11-45-23
  PPO + LiDAR, Obstacle world: RL_RSL/2026-04-14_18-44-30_finetune_dense_obstacle
"""

import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CKPT_ROOT   = "/media/parth/Elements/parth/trained_checkpoints"
REPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
WINDOW_FRAC = 0.1   # use last 10% of training for stats

RUNS = {
    "PPO + LiDAR\nEmpty world":    os.path.join(CKPT_ROOT, "RL_RSL/2026-04-14_11-45-23"),
    "PPO + LiDAR\nObstacle world": os.path.join(CKPT_ROOT, "RL_RSL/2026-04-14_18-44-30_finetune_dense_obstacle"),
}

# (tag, display name, higher_is_better)
METRICS = [
    ("Train/mean_reward",                    "Mean Episode Return (MER)",      True),
    ("Train/mean_episode_length",            "Mean Episode Length (steps)",     True),
    ("Episode_Termination/time_out",         "Timeout Rate",                    True),
    ("Episode_Termination/bad_orientation",  "Bad-Orientation Rate",            False),
    ("Episode_Termination/base_height",      "Base-Height Termination Rate",    False),
    ("Episode_Reward/track_lin_vel_xy",      "Velocity Tracking Reward",        True),
    ("Episode_Reward/flat_orientation_l2",   "Flat Orientation Penalty",        False),
    ("Episode_Reward/alive",                 "Alive Reward",                    True),
    # Obstacle-world only
    ("Episode_Termination/obstacle_collision", "Obstacle Collision Rate",       False),
    ("Episode_Reward/object_hit",            "Object-Hit Penalty",              False),
    ("Episode_Reward/lidar_proximity",       "LiDAR Proximity Penalty",         False),
]


def load_ea(run_dir: str) -> EventAccumulator:
    ef = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))[0]
    ea = EventAccumulator(ef)
    ea.Reload()
    return ea


def get_final_stats(ea: EventAccumulator, tag: str, window_frac: float):
    """Return (mean, std) over the last window_frac of logged values."""
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return None, None
    vals = np.array([e.value for e in events], dtype=float)
    n    = max(1, int(len(vals) * window_frac))
    tail = vals[-n:]
    return float(np.mean(tail)), float(np.std(tail))


# ── Collect stats ──────────────────────────────────────────────────────────────
print("Loading event files...")
eas = {name: load_ea(path) for name, path in RUNS.items()}
run_names = list(RUNS.keys())

results = {}   # metric_name → {run_name: (mean, std)}
for tag, display, hib in METRICS:
    results[display] = {}
    for name, ea in eas.items():
        m, s = get_final_stats(ea, tag, WINDOW_FRAC)
        results[display][name] = (m, s)


# ── Print table ────────────────────────────────────────────────────────────────
COL_W = 32

header = f"{'Metric':<{COL_W}}" + "".join(f"{'PPO Empty':>22}{'PPO Obstacle':>22}")
print("\n" + "=" * (COL_W + 44))
print(f"{'Metric':<{COL_W}}  {'PPO + LiDAR  Empty':>20}  {'PPO + LiDAR  Obstacle':>22}")
print("=" * (COL_W + 44))

lines = []
for (tag, display, hib), (metric_name, run_stats) in zip(METRICS, results.items()):
    row = f"{metric_name:<{COL_W}}"
    for rname in run_names:
        m, s = run_stats[rname]
        if m is None:
            cell = "      N/A       "
        else:
            cell = f"{m:+.4f} ± {s:.4f}"
        row += f"  {cell:>20}"
    print(row)
    lines.append(row)

print("=" * (COL_W + 44))


# ── Save to file ───────────────────────────────────────────────────────────────
out_path = os.path.join(REPORT_DIR, "stats.txt")
with open(out_path, "w") as f:
    f.write(f"Stats over last {WINDOW_FRAC*100:.0f}% of training\n\n")
    f.write(f"{'Metric':<{COL_W}}  {'PPO + LiDAR  Empty':>20}  {'PPO + LiDAR  Obstacle':>22}\n")
    f.write("=" * (COL_W + 44) + "\n")
    for line in lines:
        f.write(line + "\n")
    f.write("=" * (COL_W + 44) + "\n")

print(f"\nSaved to {os.path.relpath(out_path, REPORT_DIR)}")


# ── LaTeX snippet ──────────────────────────────────────────────────────────────
print("\n── LaTeX table rows ─────────────────────────────────────────────────────")
for metric_name, run_stats in results.items():
    cells = []
    for rname in run_names:
        m, s = run_stats[rname]
        if m is None:
            cells.append("N/A")
        else:
            cells.append(f"${m:+.3f} \\pm {s:.3f}$")
    print(f"  {metric_name} & {' & '.join(cells)} \\\\")
