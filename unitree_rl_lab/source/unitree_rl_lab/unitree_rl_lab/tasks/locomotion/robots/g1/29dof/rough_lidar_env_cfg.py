import math

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import ObservationsCfg as BaseObservationsCfg
from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg, RobotSceneCfg

_DOWNWARD_45_DEG_QUAT = (math.cos(math.pi / 8.0), 0.0, -math.sin(math.pi / 8.0), 0.0)


@configclass
class RoughLidarRobotSceneCfg(RobotSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=ROUGH_TERRAINS_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.15),
            rot=_DOWNWARD_45_DEG_QUAT,
        ),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-16.0, 16.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=4.0,
        ),
        max_distance=10.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class RoughLidarObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 10.0},
            clip=(0.0, 10.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(BaseObservationsCfg.CriticCfg):
        lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 10.0},
            clip=(0.0, 10.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RoughLidarRobotEnvCfg(RobotEnvCfg):
    scene: RoughLidarRobotSceneCfg = RoughLidarRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RoughLidarObservationsCfg = RoughLidarObservationsCfg()


@configclass
class RoughLidarRobotPlayEnvCfg(RobotPlayEnvCfg):
    scene: RoughLidarRobotSceneCfg = RoughLidarRobotSceneCfg(num_envs=32, env_spacing=2.5)
    observations: RoughLidarObservationsCfg = RoughLidarObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 8
