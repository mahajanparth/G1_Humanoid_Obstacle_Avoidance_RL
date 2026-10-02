# G1 Humanoid Obstacle Avoidance RL

**Obstacle-Aware Locomotion for Humanoid Robots: A Comparative Study of PPO and TD3 with LiDAR Perception**

*Parth Mahajan · Utkarsh Rai*

Reinforcement learning research for the **Unitree G1 29-DOF humanoid robot** studying obstacle-aware locomotion in simulation. We compare PPO and TD3 across two worlds (empty and obstacle-cluttered) and two observation modalities (proprioception-only vs. proprioception + LiDAR).

![Python 3.10](https://img.shields.io/badge/python-3.10-blue) ![Isaac Lab v2.2](https://img.shields.io/badge/Isaac%20Lab-v2.2-green) ![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-orange)

---

## Key Findings

- **PPO dominates TD3** in both worlds: 31% longer episodes (18.4 s vs. 14.0 s) in the empty world; obstacle-world TD3 collapses similarly.
- **LiDAR is essential for obstacle avoidance**: PPO+LiDAR reaches 94.5% timeout rate in the obstacle world vs. near-zero without it.
- **TD3 cannot leverage LiDAR**: adding 1,720-dim LiDAR to TD3 causes complete failure (0% timeout, MER = −2.12) because single-environment serial rollout cannot learn from a high-dimensional sensor under collision termination.
- **Two-stage training is critical**: zero-shot transfer of an empty-world policy into the obstacle world yields 34× shorter episodes (0.46 s vs. 16.0 s for the fine-tuned policy).

---

## Demo

https://github.com/mahajanparth/G1_Humanoid_Obstacle_Avoidance_RL/raw/main/unitree_rl_lab/report/obstacle_avoidance_dense_scene.mp4

---

## Results

| Ablation | Algorithm | World | MEL (s) ↑ | Timeout ↑ | Collision ↓ |
|---|---|---|---|---|---|
| I | PPO | Empty, no LiDAR | 18.4 ± 0.4 | 83.8% | — |
| I | TD3 | Empty, no LiDAR | 14.0 ± 0.6 | 64.8% | — |
| II | PPO | Obstacle, no LiDAR | 1.9 ± 0.3 | 0.5% | high |
| II | TD3 | Obstacle, no LiDAR | 2.1 ± 0.3 | 12.5% | high |
| III | **PPO + LiDAR** | **Empty** | **19.9 ± 0.3** | **99.8%** | — |
| III | TD3 + LiDAR | Empty | 2.0 ± 0.1 | 0.0% | — |
| IV | **PPO + LiDAR** | **Obstacle (fine-tune)** | **19.4 ± 0.3** | **94.5%** | **4.2%** |
| V | PPO + LiDAR (zero-shot) | Obstacle | 0.46 | ~0% | — |
| V | **PPO + LiDAR (fine-tuned)** | **Obstacle** | **16.0 ± 7.7** | high | low |

*MEL = Mean Episode Length; episode timeout at 20 s.*

| | |
|---|---|
| ![Episode length comparison](unitree_rl_lab/report/eval_comparison_length.png) | ![Return comparison](unitree_rl_lab/report/eval_comparison_return.png) |

---

## Paper

**Obstacle-Aware Locomotion for Humanoid Robots: A Comparative Study of PPO and TD3 with LiDAR Perception**
Parth Mahajan, Utkarsh Rai — NeurIPS 2025 format

- [PDF](unitree_rl_lab/report/paper.pdf)
- [LaTeX source](unitree_rl_lab/report/paper.tex)

---

## Trained Policies and Checkpoints

`.pt` checkpoints are stored via **Git LFS** — run `git lfs pull` after cloning.

| Policy | Checkpoint | Size | Ablation |
|---|---|---|---|
| PPO empty no-LiDAR | [checkpoints/ppo_empty_no_lidar/model_800.pt](unitree_rl_lab/checkpoints/ppo_empty_no_lidar/model_800.pt) | 9.6 MB | I baseline |
| PPO obstacle no-LiDAR | [checkpoints/ppo_obstacle_no_lidar/model_0.pt](unitree_rl_lab/checkpoints/ppo_obstacle_no_lidar/model_0.pt) | 66 MB | II |
| PPO LiDAR rough terrain | [checkpoints/ppo_lidar_rough/model_0.pt](unitree_rl_lab/checkpoints/ppo_lidar_rough/model_0.pt) | 53 MB | pre-train |
| **PPO LiDAR + obstacles (best)** | [checkpoints/ppo_lidar_obstacle_random/model_1200.pt](unitree_rl_lab/checkpoints/ppo_lidar_obstacle_random/model_1200.pt) | 164 MB | **III/IV** |
| TD3 no-LiDAR | [checkpoints/td3_obstacle_random/model_50000_steps.zip](unitree_rl_lab/checkpoints/td3_obstacle_random/model_50000_steps.zip) | 329 MB | I/II comparison |

**ONNX deploy policies** (for on-robot inference via ONNX Runtime) are in-repo:
- [`deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx`](unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx) — velocity-tracking baseline
- [`deploy/robots/g1_29dof/config/policy/mimic/dance_102/exported/policy.onnx`](unitree_rl_lab/deploy/robots/g1_29dof/config/policy/mimic/dance_102/exported/policy.onnx)
- [`deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/exported/policy.onnx`](unitree_rl_lab/deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/exported/policy.onnx)

---

## Repository Layout

```text
.
├── rsl_rl/                        local RSL-RL library (PPO)
└── unitree_rl_lab/
    ├── checkpoints/               trained model checkpoints (Git LFS)
    ├── deploy/                    C++ on-robot controllers + ONNX policies
    ├── report/
    │   ├── paper.pdf              compiled paper (20 pages)
    │   ├── paper.tex              LaTeX source (NeurIPS 2025)
    │   └── plots/                 54 training-curve plots
    ├── scripts/
    │   ├── rsl_rl/train.py        PPO training entry point
    │   ├── rsl_rl/play.py         policy playback + ONNX export
    │   ├── rsl_rl/eval.py         500-episode evaluation
    │   └── sb3/train_td3.py       TD3 training entry point
    ├── source/unitree_rl_lab/     Isaac Lab task definitions
    └── unitree_rl_lab.sh          helper script
```

---

## Setup

**Prerequisites:** Linux, Python 3.10, Conda, [Isaac Lab v2.2](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/binaries_installation.html)

```bash
git clone git@github.com:mahajanparth/G1_Humanoid_Obstacle_Avoidance_RL.git
cd G1_Humanoid_Obstacle_Avoidance_RL
git lfs pull                        # download LFS checkpoints

conda activate env_isaaclab
cd unitree_rl_lab && ./unitree_rl_lab.sh -i
conda deactivate && conda activate env_isaaclab
```

**Robot assets** (required for simulation):
```bash
# Option A: USD assets
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
git -C unitree_model lfs pull

# Option B: URDF assets
git clone https://github.com/unitreerobotics/unitree_ros.git
```

---

## Play a Checkpoint

```bash
cd unitree_rl_lab

# Best obstacle-avoidance policy (PPO + LiDAR)
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --checkpoint checkpoints/ppo_lidar_obstacle_random/model_1200.pt \
  --draw_lidar_lines

# Baseline velocity policy
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity \
  --checkpoint checkpoints/ppo_empty_no_lidar/model_800.pt
```

---

## Reproduce Paper Experiments

All commands run from `unitree_rl_lab/`. Scale settings: 4,096 parallel envs, 50,000 PPO iterations.

| Experiment | Command |
|---|---|
| PPO empty no-LiDAR | `python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --num_envs 4096 --max_iterations 50000` |
| PPO empty + LiDAR (pre-train) | `python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Empty-Lidar-Velocity --num_envs 4096 --max_iterations 50000` |
| PPO obstacle + LiDAR (fine-tune) | `python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Dense-Obstacle-Velocity --num_envs 4096 --resume --load_run <pretrain_run>` |
| TD3 empty no-LiDAR | `python scripts/sb3/train_td3.py --task Unitree-G1-29dof-Velocity --total_timesteps 5000000 --policy_delay 8 --actor_lr_scale 0.1` |
| Evaluate (500 episodes) | `python scripts/rsl_rl/eval.py --task <TASK> --num_envs 64 --num_episodes 500 --checkpoint <path>` |

### Available Tasks

```
Unitree-Go2-Velocity                         Unitree-H1-Velocity
Unitree-G1-29dof-Velocity                    Unitree-G1-29dof-Obstacle-Velocity
Unitree-G1-29dof-Rough-Lidar-Velocity        Unitree-G1-29dof-Random-Obstacle-Velocity
Unitree-G1-29dof-Empty-Lidar-Velocity        Unitree-G1-29dof-Single-Obstacle-Velocity
Unitree-G1-29dof-Dense-Obstacle-Velocity     Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval
Unitree-G1-29dof-Empty-FullLidar-Eval        Unitree-G1-29dof-Mimic-Dance-102
Unitree-G1-29dof-Mimic-Gangnanm-Style
```

```bash
./unitree_rl_lab.sh -l    # list all registered tasks
```

---

## Deploy to Hardware

C++ controllers for on-robot deployment (uses ONNX Runtime 1.22.0, bundled):

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof
mkdir -p build && cd build
cmake .. && make
```

Supported robots: `go2`, `go2w`, `b2`, `h1`, `h1_2`, `g1_23dof`, `g1_29dof`

---

## Logs and TensorBoard

```bash
# PPO logs: unitree_rl_lab/logs/rsl_rl/<experiment>/<run>/
# TD3 logs: unitree_rl_lab/logs/sb3/<experiment>/<run>/
tensorboard --logdir unitree_rl_lab/logs --port 6006
```
