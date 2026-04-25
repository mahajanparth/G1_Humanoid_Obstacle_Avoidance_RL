import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Empty-Lidar-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:EmptyLidarRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:EmptyLidarRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Single-Obstacle-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:SingleObstacleRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:SingleObstacleRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Dense-Obstacle-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:DenseObstacleRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:DenseObstacleRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Obstacle-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacle_env_cfg:ObstacleRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.obstacle_env_cfg:ObstacleRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Rough-Lidar-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_lidar_env_cfg:RoughLidarRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_lidar_env_cfg:RoughLidarRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Random-Obstacle-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:RandomObstacleRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:RandomObstacleRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Eval tasks for checkpoints trained with full 248-ray lidar (before compression).
# DenseObstacle variant: used for both obstacle-trained and empty→obstacle transfer evals.
gym.register(
    id="Unitree-G1-29dof-Dense-Obstacle-FullLidar-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:DenseObstacleRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:DenseObstacleFullLidarEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# Empty-world eval task for checkpoints trained with full 248-ray lidar.
gym.register(
    id="Unitree-G1-29dof-Empty-FullLidar-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:EmptyLidarRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.random_obstacle_env_cfg:EmptyFullLidarEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
