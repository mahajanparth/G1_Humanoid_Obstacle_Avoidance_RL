# G1 Rough Terrain + LiDAR-Style Exteroception Setup Guide

This guide uses the current official stack that best matches your project proposal:

- Simulator/task stack: **Isaac Lab**
- Robot/task: **`Isaac-Velocity-Rough-G1-v0`**
- RL library: **`rsl_rl` / PPO**
- Exteroception: **built-in rough-terrain height scanner**

Why this route:

- It is the cleanest official G1 rough-terrain task available today.
- The stock rough G1 task already includes a ray-cast height scanner, which is a practical match for the proposal's "LiDAR/raycast terrain features".
- It is more maintainable than building on legacy `legged_gym` + Isaac Gym Preview 3.

## 0. Important Prerequisite

On your current machine, `nvidia-smi` failed earlier. Do **not** continue until this works.

Run:

```bash
nvidia-smi
```

If this fails, fix the NVIDIA driver first. Isaac Lab training will not work reliably without a working GPU driver/runtime.

Also note:

- Your system Python is `3.13`, which is **not** the Python you want for Isaac Lab.
- Use a dedicated Conda environment created by `isaaclab.sh`.

## 1. Install Isaac Sim 5 and Isaac Lab

### 1.1 Download Isaac Sim 5

Download the Linux binary release of Isaac Sim 5 from NVIDIA and extract it into `~/isaacsim`.

Example:

```bash
mkdir -p ~/isaacsim
tar -xvf ~/Downloads/IsaacSim-5.0.0-linux-x86_64.tar.gz -C ~/isaacsim --strip-components=1
```

Adjust the tarball name if your downloaded file differs.

### 1.2 Clone Isaac Lab

Use a stable release tag instead of the moving `main` branch:

```bash
cd ~
git clone --branch v2.2.0 https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
```

### 1.3 Link Isaac Sim into Isaac Lab

```bash
cd ~/IsaacLab
ln -s ~/isaacsim _isaac_sim
```

### 1.4 Create the Conda environment

```bash
cd ~/IsaacLab
./isaaclab.sh --conda g1rl
conda activate g1rl
```

### 1.5 Install Isaac Lab extensions and PPO support

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh --install rsl_rl
```

## 2. Smoke-Test the Installation

### 2.1 Basic simulator test

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

The first run can take a while because Isaac Sim may build caches/shaders.

### 2.2 Train stock rough-terrain G1 once

This confirms the official task runs before you customize anything:

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 \
  --headless \
  --num_envs 1024
```

If VRAM is tight, start with:

```bash
--num_envs 512
```

If this step fails, do not proceed to task customization yet.

## 3. What Counts as "LiDAR" in This Setup

The stock rough G1 task already uses a terrain ray-caster / height scanner.

That means:

- **rough terrain** is already part of the stock task
- **exteroceptive terrain observations** are already part of the stock task

For your report, this is the cleanest interpretation:

- **Proprio-only baseline**: disable the height scanner observation
- **LiDAR-style exteroceptive policy**: keep the rough-task height scanner enabled

This is acceptable because your proposal explicitly allows either:

- LiDAR / raycast beams, or
- terrain-height samples in the robot frame

The stock task uses the second form.

## 4. Create Two Custom Tasks

You want two side-by-side tasks:

- `Isaac-Velocity-Rough-G1-Proprio-v0`
- `Isaac-Velocity-Rough-G1-Lidar-v0`

The easiest clean way is to create a small custom task package inside Isaac Lab's task tree.

### 4.1 Create a new task package directory

```bash
cd ~/IsaacLab
mkdir -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_project
```

### 4.2 Create `__init__.py`

Create:

`~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_project/__init__.py`

with:

```python
import gymnasium as gym

from ..g1 import agents

gym.register(
    id="Isaac-Velocity-Rough-G1-Proprio-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_proprio_env_cfg:G1RoughProprioEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_lidar_env_cfg:G1RoughLidarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
```

### 4.3 Create the proprio-only config

Create:

`~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_project/rough_proprio_env_cfg.py`

with:

```python
from isaaclab.utils import configclass

from ..g1.rough_env_cfg import G1RoughEnvCfg


@configclass
class G1RoughProprioEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
```

This keeps rough terrain but removes exteroceptive height-scan input from the policy.

### 4.4 Create the LiDAR-style exteroceptive config

Create:

`~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_project/rough_lidar_env_cfg.py`

with:

```python
from isaaclab.utils import configclass

from ..g1.rough_env_cfg import G1RoughEnvCfg


@configclass
class G1RoughLidarEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # The stock rough G1 task already uses a ray-cast terrain scanner.
        # These settings make that choice explicit for the project setup.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.height_scanner.pattern_cfg.resolution = 0.1
        self.scene.height_scanner.pattern_cfg.size = [1.6, 1.0]

        self.observations.policy.height_scan.clip = (-1.0, 1.0)
```

This is the recommended "LiDAR-style" version for the project because it stays close to the official rough G1 environment.

## 5. Train the Two Variants

### 5.1 Train the proprio-only baseline

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-Proprio-v0 \
  --headless \
  --num_envs 1024
```

### 5.2 Train the exteroceptive / LiDAR-style variant

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-Lidar-v0 \
  --headless \
  --num_envs 1024
```

For cleaner comparisons:

- use the same seed
- use the same number of environments
- use the same training budget
- change only the observation configuration

Example:

```bash
--seed 1
```

## 6. Visualize a Trained Policy

After training, test a policy with:

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-G1-Lidar-v0
```

Swap the task name to the proprio variant as needed.

## 7. If You Explicitly Want a Literal LiDAR

The stock rough G1 task uses a **grid-based height scanner**. If you explicitly want a **LiDAR-shaped scan**, the cleanest path is:

1. keep the stock rough-terrain task,
2. keep using Isaac Lab's `RayCasterCfg`,
3. replace the `GridPatternCfg` with `LidarPatternCfg`,
4. replace the `height_scan` observation with a custom observation that returns LiDAR ranges.

This is still a warp-based ray caster, not a full RTX LiDAR pipeline, but it is the most practical RL setup.

### 7.1 Why this is the right approach

Isaac Lab's ray caster already supports a generic LiDAR pattern.

That gives you:

- explicit beams
- configurable vertical channels
- configurable horizontal field of view
- configurable angular resolution

For RL, this is usually the right compromise between realism and training throughput.

### 7.2 Create a custom LiDAR observation helper

Create:

`~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_project/lidar_obs.py`

with:

```python
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def lidar_ranges(env, sensor_cfg: SceneEntityCfg, max_range: float) -> torch.Tensor:
    """Return normalized LiDAR ranges in [0, 1]."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    origins_w = sensor.data.pos_w.unsqueeze(1)
    hits_w = sensor.data.ray_hits_w

    ranges = torch.linalg.norm(hits_w - origins_w, dim=-1)
    no_hit = torch.isinf(hits_w).any(dim=-1)
    ranges = torch.where(no_hit, torch.full_like(ranges, max_range), ranges)

    return torch.clamp(ranges / max_range, 0.0, 1.0)
```

### 7.3 Replace the stock height scanner with a LiDAR pattern

Update the LiDAR task config:

`~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_project/rough_lidar_env_cfg.py`

to:

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from ..g1.rough_env_cfg import G1RoughEnvCfg
from .lidar_obs import lidar_ranges


@configclass
class G1RoughLidarEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        max_range = 20.0

        # Re-use the stock rough-terrain scanner slot, but turn it into a LiDAR.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.15, 0.0, 0.10))
        self.scene.height_scanner.ray_alignment = "base"
        self.scene.height_scanner.max_distance = max_range
        self.scene.height_scanner.pattern_cfg = patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=2.0,
        )
        self.scene.height_scanner.debug_vis = False

        # Replace terrain-height samples with normalized LiDAR beam ranges.
        self.observations.policy.height_scan = ObsTerm(
            func=lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "max_range": max_range},
            clip=(0.0, 1.0),
        )
```

This gives you:

- a forward-looking LiDAR-like beam pattern
- explicit range observations
- the same rough terrain and PPO training stack

### 7.4 Keep the proprio baseline unchanged

Your proprio-only task should still remove the exteroceptive term entirely:

```python
from isaaclab.utils import configclass

from ..g1.rough_env_cfg import G1RoughEnvCfg


@configclass
class G1RoughProprioEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
```

### 7.5 Train with the literal LiDAR variant

```bash
cd ~/IsaacLab
conda activate g1rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-Lidar-v0 \
  --headless \
  --num_envs 1024 \
  --seed 1
```

### 7.6 Practical tuning advice

Start small. A very dense LiDAR will slow training.

Reasonable first settings:

- `channels=16`
- `horizontal_fov_range=(-90, 90)`
- `horizontal_res=2.0`
- `max_range=10` to `20`

That keeps the observation size manageable.

### 7.7 Important caveat

Isaac Lab's ray caster currently works against **static meshes** that you specify in `mesh_prim_paths`.

For your rough-terrain locomotion project, that is fine because the ground terrain is the main thing you want to scan.

If you later want a more photorealistic or hardware-faithful LiDAR model, move to an RTX LiDAR path in Isaac Sim. That is a different setup and more expensive to train with.

## 8. Recommended Experimental Order

1. Verify `nvidia-smi` works.
2. Install Isaac Sim 5 + Isaac Lab.
3. Train the stock `Isaac-Velocity-Rough-G1-v0`.
4. Add the two custom variants above.
5. Train `Proprio` and `Lidar` with the same seed and training budget.
6. Compare return, fall rate, and robustness on rough terrain.

## 9. Common Failure Modes

- `nvidia-smi` fails:
  your driver/runtime is broken; fix this first.
- Isaac Lab launches but training crashes immediately:
  usually environment/version mismatch or missing Isaac Sim link.
- Out-of-memory on GPU:
  reduce `--num_envs` to `512` or `256`.
- Custom task name not found:
  check the package/file paths and Python syntax in `g1_project`.
- No difference between variants:
  verify `height_scan` really is disabled in the proprio task.

## 10. Minimal Command Checklist

```bash
nvidia-smi
mkdir -p ~/isaacsim
tar -xvf ~/Downloads/IsaacSim-5.0.0-linux-x86_64.tar.gz -C ~/isaacsim --strip-components=1
cd ~
git clone --branch v2.2.0 https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
ln -s ~/isaacsim _isaac_sim
./isaaclab.sh --conda g1rl
conda activate g1rl
./isaaclab.sh --install rsl_rl
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --headless --num_envs 1024
```

## Sources

- Isaac Lab installation docs: https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/binaries_installation.html
- Isaac Lab available environments: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- Official G1 rough environment registration: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py
- Official G1 rough config: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py
- Base velocity task config showing the height scanner and `height_scan` observation term: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
