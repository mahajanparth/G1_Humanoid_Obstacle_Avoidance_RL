import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import EventCfg as BaseEventCfg
from .velocity_env_cfg import ObservationsCfg as BaseObservationsCfg
from .velocity_env_cfg import RewardsCfg as BaseRewardsCfg
from .velocity_env_cfg import CurriculumCfg as BaseCurriculumCfg
from .velocity_env_cfg import TerminationsCfg as BaseTerminationsCfg
from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg, RobotSceneCfg

RANDOM_OBSTACLE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class RandomObstacleRobotSceneCfg(RobotSceneCfg):
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, -1.5, 0.9)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, 1.5, 1.1)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.5, -0.5, 1.4)),
    )
    obstacle_d = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleD",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.5, 3.4),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.65, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 2.0, 1.7)),
    )
    obstacle_e = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleE",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.3, 2.6),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.35, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, -2.0, 1.3)),
    )
    obstacle_f = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleF",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.55, 3.0),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.8, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.8, -1.8, 1.5)),
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
            "{ENV_REGEX_NS}/ObstacleD",
            "{ENV_REGEX_NS}/ObstacleE",
            "{ENV_REGEX_NS}/ObstacleF",
        ],
    )


@configclass
class EmptyLidarRobotSceneCfg(RobotSceneCfg):
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
        ],
        debug_vis=False,
    )


@configclass
class SingleObstaclePlaySceneCfg(RandomObstacleRobotSceneCfg):
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.9)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 1.6, 1.1)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, -0.6, 1.4)),
    )
    obstacle_d = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleD",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.5, 3.4),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.65, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.0, 2.0, 1.7)),
    )
    obstacle_e = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleE",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.3, 2.6),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.35, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.8, 2.5, 1.3)),
    )
    obstacle_f = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleF",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.55, 3.0),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.8, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.7, -2.3, 1.5)),
    )


@configclass
class DenseObstaclePlaySceneCfg(RandomObstacleRobotSceneCfg):
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, -1.5, 0.9)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.5, 1.5, 1.1)),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.5, -0.4, 1.4)),
    )
    obstacle_d = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleD",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.5, 3.4),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.65, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.5, 1.2, 1.7)),
    )
    obstacle_e = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleE",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 0.3, 2.6),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.35, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(7.2, -2.0, 1.3)),
    )
    obstacle_f = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleF",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.55, 3.0),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.8, 0.75)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.7, -2.3, 1.5)),
    )


@configclass
class FullLidarObservationsCfg(BaseObservationsCfg):
    """Observation config that uses the FULL 248-ray lidar (matches pre-compression checkpoints)."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 20.0},
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(BaseObservationsCfg.CriticCfg):
        lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 20.0},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RandomObstacleObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 20.0},
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(BaseObservationsCfg.CriticCfg):
        lidar = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 20.0},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RandomObstacleRewardsCfg(BaseRewardsCfg):
    object_hit = RewTerm(
        func=mdp.object_hit_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("torso_obstacle_contact")},
    )
    lidar_proximity = RewTerm(
        func=mdp.lidar_obstacle_proximity_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": 20.0, "safe_distance": 0.8},
    )


@configclass
class EmptyLidarRewardsCfg(BaseRewardsCfg):
    pass


@configclass
class ObstacleTerminationsCfg(BaseTerminationsCfg):
    obstacle_collision = DoneTerm(
        func=mdp.contact_with_object,
        params={"sensor_cfg": SceneEntityCfg("torso_obstacle_contact"), "threshold": 1.0},
    )


@configclass
class ObstacleCurriculumCfg(BaseCurriculumCfg):
    terrain_levels = None


@configclass
class RandomObstacleEventCfg(BaseEventCfg):
    randomize_obstacles = EventTerm(
        func=mdp.randomize_obstacle_course,
        mode="reset",
        params={
            "obstacle_names": ["obstacle_a", "obstacle_b", "obstacle_c", "obstacle_d", "obstacle_e", "obstacle_f"],
            "x_range": (1.5, 5.0),
            "y_range": (-2.5, 2.5),
            "min_active": 0,
            "max_active": 3,
            "min_separation": 1.25,
            "spawn_clearance": 1.5,
            "spawn_corridor_x": 1.5,
            "spawn_corridor_y": 1.0,
            "hidden_height": -10.0,
        },
    )


@configclass
class SingleObstacleEventCfg(BaseEventCfg):
    randomize_obstacles = EventTerm(
        func=mdp.randomize_obstacle_course,
        mode="reset",
        params={
            "obstacle_names": ["obstacle_a", "obstacle_b", "obstacle_c", "obstacle_d", "obstacle_e", "obstacle_f"],
            "x_range": (1.5, 5.0),
            "y_range": (-2.5, 2.5),
            "min_active": 1,
            "max_active": 1,
            "min_separation": 1.25,
            "spawn_clearance": 1.5,
            "spawn_corridor_x": 1.5,
            "spawn_corridor_y": 1.0,
            "hidden_height": -10.0,
        },
    )


@configclass
class DenseObstacleEventCfg(BaseEventCfg):
    randomize_obstacles = EventTerm(
        func=mdp.randomize_obstacle_course,
        mode="reset",
        params={
            "obstacle_names": ["obstacle_a", "obstacle_b", "obstacle_c", "obstacle_d", "obstacle_e", "obstacle_f"],
            "x_range": (1.5, 5.0),
            "y_range": (-2.5, 2.5),
            "min_active": 3,
            "max_active": 5,
            "min_separation": 1.25,
            "spawn_clearance": 1.5,
            "spawn_corridor_x": 1.5,
            "spawn_corridor_y": 1.0,
            "hidden_height": -10.0,
        },
    )


@configclass
class EmptyLidarRobotEnvCfg(RobotEnvCfg):
    scene: EmptyLidarRobotSceneCfg = EmptyLidarRobotSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: EmptyLidarRewardsCfg = EmptyLidarRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()


@configclass
class EmptyLidarRobotPlayEnvCfg(RobotPlayEnvCfg):
    scene: EmptyLidarRobotSceneCfg = EmptyLidarRobotSceneCfg(num_envs=1, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: EmptyLidarRewardsCfg = EmptyLidarRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class SingleObstacleRobotEnvCfg(RobotEnvCfg):
    scene: SingleObstaclePlaySceneCfg = SingleObstaclePlaySceneCfg(num_envs=4096, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: SingleObstacleEventCfg = SingleObstacleEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()


@configclass
class SingleObstacleRobotPlayEnvCfg(RobotPlayEnvCfg):
    scene: SingleObstaclePlaySceneCfg = SingleObstaclePlaySceneCfg(num_envs=1, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class DenseObstacleRobotEnvCfg(RobotEnvCfg):
    scene: DenseObstaclePlaySceneCfg = DenseObstaclePlaySceneCfg(num_envs=4096, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: DenseObstacleEventCfg = DenseObstacleEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()


@configclass
class DenseObstacleRobotPlayEnvCfg(RobotPlayEnvCfg):
    scene: DenseObstaclePlaySceneCfg = DenseObstaclePlaySceneCfg(num_envs=1, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class RandomObstacleRobotEnvCfg(RobotEnvCfg):
    scene: RandomObstacleRobotSceneCfg = RandomObstacleRobotSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: RandomObstacleEventCfg = RandomObstacleEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()


@configclass
class RandomObstacleRobotPlayEnvCfg(RobotPlayEnvCfg):
    scene: RandomObstacleRobotSceneCfg = RandomObstacleRobotSceneCfg(num_envs=1, env_spacing=6.0)
    observations: RandomObstacleObservationsCfg = RandomObstacleObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: RandomObstacleEventCfg = RandomObstacleEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class DenseObstacleFullLidarEvalEnvCfg(RobotPlayEnvCfg):
    """Play/eval env using full 248-ray lidar obs — matches checkpoints trained before lidar compression.

    Use this for evaluating:
      - obstacle-trained PPO+lidar checkpoint on obstacle world
      - empty-world PPO+lidar checkpoint transferred to obstacle world (Ablation V)
    """

    scene: DenseObstaclePlaySceneCfg = DenseObstaclePlaySceneCfg(num_envs=64, env_spacing=6.0)
    observations: FullLidarObservationsCfg = FullLidarObservationsCfg()
    rewards: RandomObstacleRewardsCfg = RandomObstacleRewardsCfg()
    events: DenseObstacleEventCfg = DenseObstacleEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class EmptyFullLidarEvalEnvCfg(RobotPlayEnvCfg):
    """Play/eval env for the empty-world PPO+lidar checkpoint (full 248-ray lidar obs)."""

    scene: EmptyLidarRobotSceneCfg = EmptyLidarRobotSceneCfg(num_envs=64, env_spacing=6.0)
    observations: FullLidarObservationsCfg = FullLidarObservationsCfg()
    rewards: EmptyLidarRewardsCfg = EmptyLidarRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: ObstacleCurriculumCfg = ObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
