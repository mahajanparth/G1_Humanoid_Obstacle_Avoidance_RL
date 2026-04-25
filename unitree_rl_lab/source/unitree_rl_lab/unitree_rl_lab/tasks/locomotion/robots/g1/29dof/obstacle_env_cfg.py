import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import EventCfg as BaseEventCfg, RobotEnvCfg, RobotPlayEnvCfg, RobotSceneCfg
from .random_obstacle_env_cfg import (
    ObstacleCurriculumCfg,
    ObstacleTerminationsCfg,
    RandomObstacleObservationsCfg,
    RandomObstacleRewardsCfg,
)


@configclass
class ObstacleRobotSceneCfg(RobotSceneCfg):
    """Flat terrain with three fixed static obstacles and lidar."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # z = size[2] / 2 so the bottom face sits on the ground
    obstacle_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleA",
        spawn=sim_utils.CuboidCfg(
            size=(0.35, 0.45, 1.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
                max_depenetration_velocity=10.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, -1.2, 0.9)),
    )
    obstacle_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleB",
        spawn=sim_utils.CuboidCfg(
            size=(0.45, 0.35, 2.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
                max_depenetration_velocity=10.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.7, 0.35)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 1.0, 1.1)),
    )
    obstacle_c = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleC",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.4, 2.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True,
                max_depenetration_velocity=10.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.45, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.0, -0.3, 1.4)),
    )
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.15)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=6.0,
        ),
        max_distance=20.0,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/.*/Obstacle.*", track_mesh_transforms=True),
        ],
        debug_vis=False,
    )
    torso_obstacle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/ObstacleA",
            "{ENV_REGEX_NS}/ObstacleB",
            "{ENV_REGEX_NS}/ObstacleC",
        ],
    )


@configclass
class ObstacleRobotEnvCfg(RobotEnvCfg):
    scene: ObstacleRobotSceneCfg = ObstacleRobotSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()


@configclass
class ObstacleRobotPlayEnvCfg(RobotPlayEnvCfg):
    scene: ObstacleRobotSceneCfg = ObstacleRobotSceneCfg(num_envs=1, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
