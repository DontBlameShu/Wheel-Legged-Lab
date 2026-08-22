import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import WHEEL_LEGGED_ASSET_DIR

##
# Configuration
##
WHEEL_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(WHEEL_LEGGED_ASSET_DIR / "urdf" / "COD_2026_Balance_2_0_floating.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    articulation_root_prim_path="/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.345),
        joint_pos={
            "Left_rear_joint": 0.347823,      # 髋关节直立
            "Right_rear_joint": -0.313007,
            "Left_rear_child1_joint": 0.410068,      # 膝关节微屈，保持平衡姿态
            "Right_rear_child1_joint": -0.332394,
            "Left_Wheel_joint": 0.0,  # 轮子初始角度无所谓
            "Right_Wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部关节：使用力矩控制
        "legs": DCMotorCfg(
            joint_names_expr=["Left_rear_joint", "Left_rear_child1_joint", "Right_rear_joint", "Right_rear_child1_joint"],
            effort_limit=30.0,          # 根据 URDF 中的 effort
            saturation_effort=30.0,
            velocity_limit=30.0,        # rad/s
            # 全部为0,做力矩控制：
            stiffness=0.0,             # 位置增益
            damping=0.0,                # 速度增益
            friction=0.0,
        ),
        # 轮子：使用力矩控制
        "wheels": DCMotorCfg(
            joint_names_expr=["Left_Wheel_joint", "Right_Wheel_joint"],
            effort_limit=5.0,           # URDF 中 effort=5
            saturation_effort=5.0,
            velocity_limit=1000.0,      # 轮子允许高速
            # 全部为0,做力矩控制：
            stiffness=0.0,              # 如果做速度控制，stiffness 设 0，用 damping 作比例增益
            damping=0.0,                # 速度控制的比例系数
            friction=0.0,
        ),
    },
)
