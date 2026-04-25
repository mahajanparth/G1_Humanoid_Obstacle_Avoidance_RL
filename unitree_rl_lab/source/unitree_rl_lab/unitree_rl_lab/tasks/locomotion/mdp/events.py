from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_obstacle_course(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    obstacle_names: list[str],
    x_range: tuple[float, float] = (1.5, 6.5),
    y_range: tuple[float, float] = (-3.0, 3.0),
    placement_zones: list[dict[str, tuple[float, float]]] | None = None,
    min_active: int = 1,
    max_active: int | None = None,
    min_separation: float = 1.25,
    spawn_clearance: float = 1.75,
    spawn_corridor_x: float = 3.0,
    spawn_corridor_y: float = 1.5,
    hidden_height: float = -10.0,
):
    """Randomize obstacle count and placement with a minimum separation on reset."""
    if max_active is None:
        max_active = len(obstacle_names)
    max_active = min(max_active, len(obstacle_names))
    min_active = max(0, min(min_active, max_active))

    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)
    env_origins = env.scene.env_origins[env_ids]
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    obstacle_assets: list[RigidObject] = [env.scene[name] for name in obstacle_names]
    active_counts = torch.randint(min_active, max_active + 1, (len(env_ids),), device=device)
    if placement_zones is not None and len(placement_zones) != len(obstacle_names):
        raise ValueError("placement_zones must have the same length as obstacle_names.")

    def _valid_spawn_region(candidate_xy: torch.Tensor) -> torch.Tensor:
        radial_ok = torch.linalg.norm(candidate_xy, dim=1) >= spawn_clearance
        corridor_ok = ~((candidate_xy[:, 0] <= spawn_corridor_x) & (torch.abs(candidate_xy[:, 1]) <= spawn_corridor_y))
        return radial_ok & corridor_ok

    sampled_positions = []
    for obstacle_idx in range(len(obstacle_names)):
        positions_xy = torch.empty((len(env_ids), 2), device=device)
        if obstacle_idx == 0:
            valid_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=device)
            for _ in range(32):
                candidate_xy = torch.empty((len(env_ids), 2), device=device)
                zone_x_range = x_range if placement_zones is None else placement_zones[obstacle_idx]["x_range"]
                zone_y_range = y_range if placement_zones is None else placement_zones[obstacle_idx]["y_range"]
                candidate_xy[:, 0] = torch.empty(len(env_ids), device=device).uniform_(*zone_x_range)
                candidate_xy[:, 1] = torch.empty(len(env_ids), device=device).uniform_(*zone_y_range)
                candidate_valid = _valid_spawn_region(candidate_xy)
                update_mask = (~valid_mask) & candidate_valid
                positions_xy[update_mask] = candidate_xy[update_mask]
                valid_mask |= candidate_valid
                if bool(torch.all(valid_mask)):
                    break
            if not bool(torch.all(valid_mask)):
                fallback_xy = torch.zeros((len(env_ids), 2), device=device)
                fallback_xy[:, 0] = x_range[1]
                positions_xy[~valid_mask] = fallback_xy[~valid_mask]
        else:
            valid_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=device)
            for _ in range(32):
                candidate_xy = torch.empty((len(env_ids), 2), device=device)
                zone_x_range = x_range if placement_zones is None else placement_zones[obstacle_idx]["x_range"]
                zone_y_range = y_range if placement_zones is None else placement_zones[obstacle_idx]["y_range"]
                candidate_xy[:, 0] = torch.empty(len(env_ids), device=device).uniform_(*zone_x_range)
                candidate_xy[:, 1] = torch.empty(len(env_ids), device=device).uniform_(*zone_y_range)
                candidate_valid = _valid_spawn_region(candidate_xy)
                for prev_xy in sampled_positions:
                    candidate_valid &= torch.linalg.norm(candidate_xy - prev_xy, dim=1) >= min_separation
                update_mask = (~valid_mask) & candidate_valid
                positions_xy[update_mask] = candidate_xy[update_mask]
                valid_mask |= candidate_valid
                if bool(torch.all(valid_mask)):
                    break
            if not bool(torch.all(valid_mask)):
                fallback_xy = sampled_positions[-1].clone()
                fallback_xy[:, 0] = torch.clamp(fallback_xy[:, 0] + min_separation, x_range[0], x_range[1])
                positions_xy[~valid_mask] = fallback_xy[~valid_mask]
        sampled_positions.append(positions_xy)

    for obstacle_idx, (obstacle_name, obstacle_asset) in enumerate(zip(obstacle_names, obstacle_assets, strict=True)):
        root_state = obstacle_asset.data.default_root_state[env_ids].clone()
        active_mask = active_counts > obstacle_idx

        root_state[:, 0:2] = env_origins[:, 0:2] + sampled_positions[obstacle_idx]
        root_state[:, 2] = root_state[:, 2] + env_origins[:, 2]
        root_state[:, 3:7] = identity_quat.unsqueeze(0).expand(len(env_ids), -1)
        root_state[:, 7:13] = 0.0

        if not bool(torch.all(active_mask)):
            root_state[~active_mask, 0] = env_origins[~active_mask, 0]
            root_state[~active_mask, 1] = env_origins[~active_mask, 1]
            root_state[~active_mask, 2] = env_origins[~active_mask, 2] + hidden_height

        obstacle_asset.write_root_state_to_sim(root_state, env_ids=env_ids)
