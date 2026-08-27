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
        # 腿部关节：实机单电机额定 12 N·m、峰值 25 N·m。
        # VMC 最终映射到四个实际电机，所以峰值限制应逐电机施加，
        # 而不是把一条腿两个电机的 50 N·m 合并成单一关节限幅。
        "legs": DCMotorCfg(
            joint_names_expr=["Left_rear_joint", "Left_rear_child1_joint", "Right_rear_joint", "Right_rear_child1_joint"],
            effort_limit=25.0,
            saturation_effort=25.0,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
        # 轮子：DJI M3508 + 16.875:1 减速箱。
        # 策略直接输出轮轴扭矩；4 N·m 作为短时峰值硬限制。
        "wheels": DCMotorCfg(
            joint_names_expr=["Left_Wheel_joint", "Right_Wheel_joint"],
            effort_limit=4.0,
            saturation_effort=4.0,
            velocity_limit=57.4,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
)
