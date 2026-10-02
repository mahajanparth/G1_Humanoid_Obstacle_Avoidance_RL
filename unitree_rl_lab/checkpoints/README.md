# Trained Checkpoints

All `.pt` files are stored via **Git LFS**. Run `git lfs pull` after cloning to download them.

## PPO Checkpoints (RSL-RL)

### `ppo_empty_no_lidar/model_800.pt` — 9.6 MB
- **Task:** `Unitree-G1-29dof-Velocity` (empty flat world, no LiDAR)
- **Ablation I baseline.** Proprioceptive-only policy trained for 800 iterations.
- **Results:** MEL = 18.4 ± 0.4 s, Timeout = 83.8%, Vel. tracking = 1.281

### `ppo_obstacle_no_lidar/model_0.pt` — 66 MB
- **Task:** `Unitree-G1-29dof-Obstacle-Velocity` (obstacle world, no LiDAR)
- **Ablation II.** Policy trained in obstacle world without LiDAR perception.
- **Results:** MEL = 1.9 ± 0.3 s (frequent collisions without anticipatory sensing)

### `ppo_lidar_rough/model_0.pt` — 53 MB
- **Task:** `Unitree-G1-29dof-Rough-Lidar-Velocity` (rough terrain + LiDAR)
- Intermediate pre-training run with LiDAR on rough terrain.

### `ppo_lidar_obstacle_random/model_1200.pt` — 164 MB
- **Task:** `Unitree-G1-29dof-Random-Obstacle-Velocity` (random obstacles + LiDAR)
- **Ablations III/IV — best obstacle-avoidance policy.**
  Trained for 1200 iterations with LiDAR (248 rays, 180° H-FOV) and 0–3 random obstacles per episode.
- **Results:** MER = 22.45 ± 1.01, MEL = 19.4 s, Timeout = 94.5%, Collision rate = 4.2%

## TD3 Checkpoint (Stable-Baselines3)

The TD3 checkpoint (`model_50000_steps.zip`, 329 MB) is too large for Git LFS free tier and is distributed via **GitHub Releases**:

[Download from GitHub Releases →](../../releases/tag/v1.0-checkpoints)

- **Task:** `Unitree-G1-29dof-Random-Obstacle-Velocity`
- **Ablations I/II comparison.** TD3 trained for 50,000 steps.
- TD3+LiDAR runs are omitted — they represent a failure mode (MER = −2.12, 0% timeout).

## Loading a Checkpoint

```bash
# PPO (RSL-RL)
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --checkpoint unitree_rl_lab/checkpoints/ppo_lidar_obstacle_random/model_1200.pt

# TD3 (SB3) — after downloading from GitHub Releases
python scripts/sb3/play_td3.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --checkpoint /path/to/model_50000_steps.zip
```
