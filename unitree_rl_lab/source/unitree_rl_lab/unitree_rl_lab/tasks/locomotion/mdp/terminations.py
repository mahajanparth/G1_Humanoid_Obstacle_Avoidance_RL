from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def contact_with_object(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when contact force against filtered obstacle bodies exceeds threshold.

    Uses force_matrix_w (filtered to filter_prim_paths_expr targets) so only
    obstacle contacts trigger termination, not ground or self contacts.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # force_matrix_w: [num_envs, num_bodies, history_length, 3]
    contact_forces = torch.linalg.norm(contact_sensor.data.force_matrix_w, dim=-1)
    return torch.any(contact_forces > threshold, dim=(1, 2))
