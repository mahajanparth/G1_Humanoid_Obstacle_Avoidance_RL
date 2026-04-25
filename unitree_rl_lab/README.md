# Unitree RL Lab

Isaac Lab environments, training scripts, and deploy code for Unitree robots, including G1 locomotion and obstacle-avoidance tasks.

## Prerequisites

Install Isaac Lab first, then use the same Conda environment for this package.

Expected baseline from the code:

- Linux
- Python 3.10
- Conda
- Isaac Lab already installed

## Install

From this directory:

```bash
conda activate env_isaaclab
./unitree_rl_lab.sh -i
```

Then restart the shell or reactivate the Conda environment.

The install helper does three things:

- installs `source/unitree_rl_lab` in editable mode
- adds Conda activation hooks
- enables task-name autocompletion

## Asset Setup

Place one of these folders next to this project so the code can resolve robot assets automatically:

### USD assets

```bash
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
git -C unitree_model lfs pull
```

### URDF assets

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git
```

## List Tasks

```bash
./unitree_rl_lab.sh -l
```

Current registered tasks include:

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

These commands correspond to the experiments in:

`Obstacle-Aware Locomotion for Humanoid Robots: A Comparative Study of PPO and TD3 with LiDAR Perception`

### Reproducible directly from the current task registry

PPO empty world, no LiDAR:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name paper_ppo_empty_no_lidar
```

TD3 empty world, no LiDAR:

```bash
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

PPO empty world with LiDAR pre-training:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Empty-Lidar-Velocity \
  --num_envs 4096 \
  --max_iterations 50000 \
  --experiment_name unitree_g1_29dof_empty_lidar_velocity \
  --run_name pretrain_empty_lidar
```

PPO dense-obstacle LiDAR fine-tuning from the empty-world checkpoint:

```bash
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

### Evaluation commands used for paper-style reporting

Evaluate PPO checkpoint over 500 episodes:

```bash
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/best_model.pt
```

Evaluate TD3 checkpoint over 500 episodes:

```bash
python scripts/sb3/eval_td3.py \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/checkpoint.zip
```

Transfer comparison for Ablation V:

```bash
python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/empty_world_pretrain_best_model.pt \
  --out logs/paper_eval_empty_transfer.json

python scripts/rsl_rl/eval.py \
  --task Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval \
  --num_envs 64 \
  --num_episodes 500 \
  --checkpoint /absolute/path/to/obstacle_finetune_best_model.pt \
  --out logs/paper_eval_obstacle_finetuned.json
```

### Current limitation

The obstacle-world no-LiDAR ablation from the paper is not currently exposed as a separate registered task in this checkout. The obstacle tasks in the current registry use LiDAR observations.

## RSL-RL Training

Train with the helper:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Obstacle-Velocity
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Rough-Lidar-Velocity
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Random-Obstacle-Velocity
```

Equivalent direct command:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Random-Obstacle-Velocity
```

Useful flags:

- `--num_envs`
- `--seed`
- `--max_iterations`
- `--log_root`
- `--video`
- `--draw_lidar_lines`
- `--headless`

## Play a Checkpoint

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

`play.py` also exports:

- `policy.pt`
- `policy.onnx`

into an `exported/` folder in the run directory.

## TD3 Training

Install SB3 if needed:

```bash
pip install "stable-baselines3>=2.6" rich tqdm
```

Run TD3:

```bash
./unitree_rl_lab.sh -d --task Unitree-G1-29dof-Random-Obstacle-Velocity --num_envs 1 --headless
```

Equivalent direct command:

```bash
python scripts/sb3/train_td3.py \
  --task Unitree-G1-29dof-Random-Obstacle-Velocity \
  --num_envs 1 \
  --headless
```

Useful TD3 flags:

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

## Logs

Default output folders:

- `logs/rsl_rl/`
- `logs/sb3/`

View logs with TensorBoard:

```bash
tensorboard --logdir logs --port 6006
```

## Helper Commands

```bash
./unitree_rl_lab.sh -i
./unitree_rl_lab.sh -l
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Random-Obstacle-Velocity
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Random-Obstacle-Velocity
./unitree_rl_lab.sh -d --task Unitree-G1-29dof-Random-Obstacle-Velocity --num_envs 1 --headless
```

## Deploy

Deploy code lives in:

```text
deploy/
```

For G1-29dof:

```bash
cd deploy/robots/g1_29dof
mkdir -p build
cd build
cmake ..
make
```
