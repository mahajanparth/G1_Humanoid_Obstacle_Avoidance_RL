from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def lidar_ranges(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"), max_distance: float = 20.0
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_origin = sensor.data.pos_w.unsqueeze(1)
    ranges = torch.linalg.norm(sensor.data.ray_hits_w - sensor_origin, dim=-1)
    ranges = torch.where(torch.isfinite(ranges), ranges, torch.full_like(ranges, max_distance))
    return torch.clamp(ranges, 0.0, max_distance)


def lidar_ranges_compressed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 20.0,
    num_sectors: int = 8,
) -> torch.Tensor:
    """Compressed lidar: minimum distance per horizontal sector (num_sectors values).

    Reduces 248-dim lidar to `num_sectors` dims by splitting rays into equal
    horizontal sectors and taking the minimum range in each. Much lower variance
    for the TD3 critic while still conveying obstacle proximity and direction.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_origin = sensor.data.pos_w.unsqueeze(1)
    ranges = torch.linalg.norm(sensor.data.ray_hits_w - sensor_origin, dim=-1)
    ranges = torch.where(torch.isfinite(ranges), ranges, torch.full_like(ranges, max_distance))
    ranges = torch.clamp(ranges, 0.0, max_distance) / max_distance  # [num_envs, num_rays]

    num_rays = ranges.shape[1]
    rays_per_sector = num_rays // num_sectors
    # Trim to exact multiple then reshape → [num_envs, num_sectors, rays_per_sector]
    ranges = ranges[:, : rays_per_sector * num_sectors]
    ranges = ranges.reshape(ranges.shape[0], num_sectors, rays_per_sector)
    return ranges.min(dim=-1).values  # [num_envs, num_sectors]
