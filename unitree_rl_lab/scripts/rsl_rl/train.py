# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help="Root directory for training logs. Defaults to logs/rsl_rl inside the repo.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--draw_lidar_lines",
    action="store_true",
    default=False,
    help="Draw debug line segments for lidar rays during training when a lidar sensor is available.",
)
parser.add_argument(
    "--draw_lidar_envs",
    type=int,
    default=8,
    help="Number of environments to visualize lidar lines for when --draw_lidar_lines is enabled.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import statistics
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

try:
    from omni.debugdraw import get_debug_draw_interface as _get_debug_draw_interface
except ImportError:
    _get_debug_draw_interface = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_LIDAR_LINE_COLOR = 0xFF00FFFF


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    if args_cli.log_root is not None:
        log_root_path = os.path.join(args_cli.log_root, "rsl_rl", agent_cfg.experiment_name)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    should_draw_lidar_lines = args_cli.draw_lidar_lines and not args_cli.headless and _get_debug_draw_interface is not None
    debug_draw = _get_debug_draw_interface() if should_draw_lidar_lines else None
    if debug_draw is not None:
        original_step = env.step

        def _step_with_lidar_debug(*step_args, **step_kwargs):
            step_output = original_step(*step_args, **step_kwargs)
            lidar_sensor = env.unwrapped.scene.sensors.get("lidar")
            if lidar_sensor is not None:
                num_envs_to_draw = min(args_cli.draw_lidar_envs, lidar_sensor.data.pos_w.shape[0])
                for env_id in range(num_envs_to_draw):
                    ray_hits = lidar_sensor.data.ray_hits_w[env_id]
                    sensor_origin = lidar_sensor.data.pos_w[env_id]
                    num_rays = ray_hits.shape[0]
                    stride = max(1, num_rays // 64)
                    sampled_hits = ray_hits[::stride]
                    start = tuple(sensor_origin.tolist())
                    for hit in sampled_hits:
                        end = tuple(hit.tolist())
                        debug_draw.draw_line(start, _LIDAR_LINE_COLOR, 2.0, end, _LIDAR_LINE_COLOR, 2.0)
            return step_output

        env.step = _step_with_lidar_debug

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.cfg["save_interval"] = agent_cfg.max_iterations + 1
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    best_checkpoint_path = os.path.join(log_dir, "best_model.pt")
    best_mean_reward = {"value": float("-inf")}
    original_logger_log = runner.logger.log

    def _log_with_best_checkpoint(*args, **kwargs):
        original_logger_log(*args, **kwargs)
        if len(runner.logger.rewbuffer) == 0:
            return
        mean_reward = statistics.mean(runner.logger.rewbuffer)
        if mean_reward > best_mean_reward["value"]:
            best_mean_reward["value"] = mean_reward
            runner.save(best_checkpoint_path, infos={"best_mean_reward": mean_reward})
            iter_checkpoint_path = os.path.join(log_dir, f"best_model_{runner.current_learning_iteration}.pt")
            runner.save(iter_checkpoint_path, infos={"best_mean_reward": mean_reward})

    runner.logger.log = _log_with_best_checkpoint

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    for checkpoint_name in os.listdir(log_dir):
        if checkpoint_name.startswith("model_") and checkpoint_name.endswith(".pt"):
            checkpoint_path = os.path.join(log_dir, checkpoint_name)
            if checkpoint_path != best_checkpoint_path:
                os.remove(checkpoint_path)
            # keep best_model_*.pt files (iteration-stamped best snapshots)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
