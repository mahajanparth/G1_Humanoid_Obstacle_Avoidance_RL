# G1 Humanoid Obstacle Avoidance RL

Reinforcement learning project for Unitree robots, centered on G1 locomotion and obstacle-avoidance tasks built on top of Isaac Lab.

This repository contains:

- `unitree_rl_lab/`: the Isaac Lab task package, training scripts, helper shell script, reports, and deploy code
- `rsl_rl/`: a local copy of the RSL-RL library used by the PPO training pipeline

## Repository Layout

```text
.
├── README.md
├── rsl_rl/
└── unitree_rl_lab/
    ├── unitree_rl_lab.sh
    ├── scripts/
    │   ├── list_envs.py
    │   ├── rsl_rl/
    │   │   ├── train.py
    │   │   └── play.py
    │   └── sb3/
    │       └── train_td3.py
    ├── source/unitree_rl_lab/
    └── deploy/
```

## Prerequisites

Before installing this repo, make sure you already have:

- Linux
- Python 3.10
- Conda
- Isaac Lab installed in a Conda environment

This code expects to run inside the same Conda environment that Isaac Lab uses.

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd G1_Humanoid_Obstacle_Avoidance_RL
```

### 2. Activate your Isaac Lab environment

```bash
conda activate env_isaaclab
```

### 3. Install this package

From the repo root:

```bash
cd unitree_rl_lab
./unitree_rl_lab.sh -i
```

What `./unitree_rl_lab.sh -i` does from the code:

- installs `source/unitree_rl_lab` in editable mode
- sets up Conda activation hooks
- enables CLI autocompletion for task names

After install, restart the shell or reactivate the Conda environment:

```bash
conda deactivate
conda activate env_isaaclab
```

## Robot Assets

The tasks expect Unitree robot description assets to exist locally.

### Option A: `unitree_model/` USD assets

Clone the asset repo next to this project root:

```bash
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
git -C unitree_model lfs pull
```

### Option B: `unitree_ros/` URDF assets

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git
```

The code resolves `unitree_model/` or `unitree_ros/` relative to the project root when those folders are present.

## List Available Tasks

From `unitree_rl_lab/`:

```bash
./unitree_rl_lab.sh -l
```

The code currently registers these Unitree tasks:

- `Unitree-Go2-Velocity`
- `Unitree-H1-Velocity`
- `Unitree-G1-29dof-Velocity`
- `Unitree-G1-29dof-Obstacle-Velocity`
- `Unitree-G1-29dof-Rough-Lidar-Velocity`
- `Unitree-G1-29dof-Random-Obstacle-Velocity`
- `Unitree-G1-29dof-Empty-Lidar-Velocity`
- `Unitree-G1-29dof-Single-Obstacle-Velocity`
- `Unitree-G1-29dof-Dense-Obstacle-Velocity`
- `Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval`
- `Unitree-G1-29dof-Empty-FullLidar-Eval`
- `Unitree-G1-29dof-Mimic-Dance-102`
- `Unitree-G1-29dof-Mimic-Gangnanm-Style`

## Paper Experiment Commands

This section maps the experiments from:

`Obstacle-Aware Locomotion for Humanoid Robots: A Comparative Study of PPO and TD3 with LiDAR Perception`

to the runnable commands in this repository.

Paper-scale settings from the code and report:

- PPO uses `4096` parallel environments by default in the G1 tasks
- PPO max training length is `50000` iterations
- TD3 evaluation in the paper uses `500` episodes
- PPO checkpoint evaluation also uses `500` episodes

Important note about reproducibility:

- The current repo directly exposes the empty-world no-LiDAR task and the LiDAR obstacle tasks.
- The current repo does not expose a separate registered CLI task for the paper's obstacle-world no-LiDAR ablation.
- Because of that, the exact no-LiDAR obstacle ablation from the paper is not a one-command rerun in the current task registry.

### Experiment Matrix

| Paper setting | Algorithm | Current task | Script |
|---|---|---|---|
| Empty world, no LiDAR | PPO | `Unitree-G1-29dof-Velocity` | `scripts/rsl_rl/train.py` |
| Empty world, no LiDAR | TD3 | `Unitree-G1-29dof-Velocity` | `scripts/sb3/train_td3.py` |
| Empty world, LiDAR | PPO | `Unitree-G1-29dof-Empty-Lidar-Velocity` | `scripts/rsl_rl/train.py` |
| Dense obstacle world, LiDAR | PPO fine-tune | `Unitree-G1-29dof-Dense-Obstacle-Velocity` | `scripts/rsl_rl/train.py` |
| Empty-world transfer eval on dense obstacles | PPO eval | `Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval` | `scripts/rsl_rl/eval.py` |
| Obstacle-trained eval on dense obstacles | PPO eval | `Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval` | `scripts/rsl_rl/eval.py` |
| Empty-world eval with full LiDAR | PPO eval | `Unitree-G1-29dof-Empty-FullLidar-Eval` | `scripts/rsl_rl/eval.py` |

### 1. PPO: Empty World, No LiDAR

Train:

```bash
cd unitree_rl_lab
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name paper_ppo_empty_no_lidar
```

Evaluate for 500 episodes:

```bash
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/best_model.pt
```

Play the trained checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity \
  --checkpoint /absolute/path/to/best_model.pt
```

### 2. TD3: Empty World, No LiDAR

Train:

```bash
cd unitree_rl_lab
python scripts/sb3/train_td3.py \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 4096 \
  --headless \
  --total_timesteps 5000000 \
  --learning_rate 1e-4 \
  --actor_lr_scale 0.1 \
  --policy_delay 8 \
  --max_grad_norm 1.0 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --learning_starts 10000 \
  --gradient_steps 4 \
  --tau 0.005 \
  --gamma 0.99 \
  --seed 42 \
  --run_name paper_td3_empty_no_lidar
```

Evaluate for 500 episodes:

```bash
python scripts/sb3/eval_td3.py \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/checkpoint.zip
```

Play the trained checkpoint:

```bash
python scripts/sb3/play_td3.py \
  --task Unitree-G1-29dof-Velocity \
  --checkpoint /absolute/path/to/checkpoint.zip
```

### 3. PPO: Empty World, LiDAR Pre-Training

This is the Stage 1 pre-training run used before obstacle fine-tuning.

Train:

```bash
cd unitree_rl_lab
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Empty-Lidar-Velocity \
  --num_envs 4096 \
  --max_iterations 50000 \
  --experiment_name unitree_g1_29dof_empty_lidar_velocity \
  --run_name pretrain_empty_lidar
```

Evaluate in the empty world:

```bash
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Empty-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/best_model.pt
```

### 4. PPO: Dense Obstacle World, LiDAR Fine-Tuning

This is the Stage 2 run for the paper's obstacle-aware PPO policy.

Important:

- `--resume` and `--load_run` load the empty-world checkpoint first
- `--experiment_name unitree_g1_29dof_empty_lidar_velocity` keeps the fine-tune run under the same experiment family as the pretrain run

Train:

```bash
cd unitree_rl_lab
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Dense-Obstacle-Velocity \
  --num_envs 4096 \
  --max_iterations 50000 \
  --experiment_name unitree_g1_29dof_empty_lidar_velocity \
  --resume \
  --load_run <pretrain_run_folder_name> \
  --run_name finetune_dense_obstacle
```

Example `load_run` value:

```text
2026-04-14_11-45-23_pretrain_empty_lidar
```

Evaluate the fine-tuned obstacle policy on the dense obstacle eval task:

```bash
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/best_model.pt
```

Play the fine-tuned obstacle policy:

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --checkpoint /absolute/path/to/best_model.pt
```

### 5. Ablation V: Empty-World Transfer vs Obstacle Fine-Tuned Policy

The paper compares:

- the empty-world PPO+LiDAR checkpoint evaluated directly in the dense obstacle world
- the obstacle fine-tuned PPO+LiDAR checkpoint evaluated in the same dense obstacle world

Evaluate the empty-world checkpoint in the dense obstacle eval task:

```bash
cd unitree_rl_lab
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/empty_world_pretrain_best_model.pt \
  --out logs/paper_eval_empty_transfer.json
```

Evaluate the fine-tuned obstacle checkpoint in the same task:

```bash
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/obstacle_finetune_best_model.pt \
  --out logs/paper_eval_obstacle_finetuned.json
```

### 6. LiDAR PPO Sanity-Check Playback

To visually inspect the paper-style LiDAR policy:

```bash
cd unitree_rl_lab
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --checkpoint /absolute/path/to/best_model.pt \
  --draw_lidar_lines
```

### 7. Obstacle-World No-LiDAR Ablation Note

The paper includes PPO-vs-TD3 obstacle-world results without LiDAR.

In the current repo state:

- `Unitree-G1-29dof-Obstacle-Velocity`
- `Unitree-G1-29dof-Random-Obstacle-Velocity`
- `Unitree-G1-29dof-Dense-Obstacle-Velocity`

all use LiDAR-based observation configs in code. So the exact no-LiDAR obstacle ablation is not currently exposed as a registered task command in this checkout.

If you want, the next step I can do is add a dedicated no-LiDAR obstacle task to the codebase so the README can include exact rerun commands for every ablation in the paper.

## Training Commands

All commands below are run from:

```bash
cd unitree_rl_lab
```

### PPO / RSL-RL training

Base G1 velocity task:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
```

Obstacle world:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Obstacle-Velocity
```

Rough terrain with lidar:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Rough-Lidar-Velocity
```

Random obstacle avoidance:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Random-Obstacle-Velocity
```

Equivalent direct Python command:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Random-Obstacle-Velocity
```

Useful flags supported by `scripts/rsl_rl/train.py`:

- `--num_envs`
- `--seed`
- `--max_iterations`
- `--log_root`
- `--video`
- `--draw_lidar_lines`
- `--headless`
- `--device`

Example:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --num_envs 64 \
  --max_iterations 2000 \
  --seed 42
```

## Play / Evaluate a Trained Policy

Play the latest checkpoint for a task:

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Random-Obstacle-Velocity
```

Play a specific checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --checkpoint /absolute/path/to/model.pt
```

Useful flags supported by `scripts/rsl_rl/play.py`:

- `--checkpoint`
- `--num_envs`
- `--video`
- `--real-time`
- `--draw_lidar_lines`
- `--device`

When `play.py` runs, it also exports the loaded policy to:

- `exported/policy.pt`
- `exported/policy.onnx`

inside the checkpoint run directory.

## TD3 Training

The repo also includes a Stable-Baselines3 TD3 training entry point:

```bash
python scripts/sb3/train_td3.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --num_envs 1 \
  --headless
```

Or through the helper:

```bash
./unitree_rl_lab.sh -d --task Unitree-G1-29dof-Random-Obstacle-Velocity --num_envs 1 --headless
```

If SB3 is not installed in your Isaac Lab environment:

```bash
pip install "stable-baselines3>=2.6" rich tqdm
```

Useful TD3 flags from the code:

- `--total_timesteps`
- `--learning_rate`
- `--lr_schedule`
- `--lr_final`
- `--buffer_size`
- `--batch_size`
- `--learning_starts`
- `--train_freq`
- `--gradient_steps`
- `--policy_delay`
- `--actor_lr_scale`
- `--tau`
- `--gamma`
- `--save_freq`
- `--log_root`

Example:

```bash
python scripts/sb3/train_td3.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --num_envs 1 \
  --headless \
  --total_timesteps 1000000 \
  --learning_rate 3e-4 \
  --buffer_size 1000000 \
  --batch_size 256
```

## Logs

Default log directories from the code are:

- PPO / RSL-RL: `unitree_rl_lab/logs/rsl_rl/<experiment_name>/`
- TD3 / SB3: `unitree_rl_lab/logs/sb3/<experiment_name>/`

Start TensorBoard from `unitree_rl_lab/`:

```bash
tensorboard --logdir logs --port 6006
```

## Helper Script Commands

From `unitree_rl_lab/`:

```bash
./unitree_rl_lab.sh -h
```

Available shortcuts:

- `./unitree_rl_lab.sh -i` install package
- `./unitree_rl_lab.sh -l` list tasks
- `./unitree_rl_lab.sh -t --task <TASK>` train with RSL-RL
- `./unitree_rl_lab.sh -p --task <TASK>` play a trained policy
- `./unitree_rl_lab.sh -d --task <TASK>` train with SB3 TD3

## Deploy

Deployment-related code is under:

```text
unitree_rl_lab/deploy/
```

Robot-specific controllers are available for:

- `go2`
- `go2w`
- `b2`
- `h1`
- `h1_2`
- `g1_23dof`
- `g1_29dof`

For G1-29dof, the deploy controller entry point is built from:

```text
unitree_rl_lab/deploy/robots/g1_29dof/
```

Typical build flow:

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof
mkdir -p build
cd build
cmake ..
make
```

## Notes

- The helper script assumes a Conda environment is already activated.
- Several older commands in previous docs used stale task names. The commands in this README were checked against the currently registered tasks in the code.
- If you are only working on the Unitree package, the most relevant README is also available at `unitree_rl_lab/README.md`.
