from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "fudan_infantry_v4"


@configclass
class FudanPlaneEnvCfg(DirectRLEnvCfg):
    # Original plane timing / spaces
    decimation = 2
    episode_length_s = 20.0
    action_space = 6
    observation_space = 25
    state_space = 141

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            min_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=3.0,
        replicate_physics=True,
    )

    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_DIR / "urdf" / "infantry_V4_increase.usd"),
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
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        articulation_root_prim_path="/base_link",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "lf0_Joint": 0.797,
                "lf1_Joint": -0.168,
                "l_wheel_Joint": 0.0,
                "rf0_Joint": -0.797,
                "rf1_Joint": 0.168,
                "r_wheel_Joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.97,
        actuators={
            # PD is NOT implemented here. The original environment computes
            # the six torques explicitly in _apply_action().
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    "lf0_Joint",
                    "lf1_Joint",
                    "rf0_Joint",
                    "rf1_Joint",
                ],
                effort_limit=50.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "l_wheel_Joint",
                    "r_wheel_Joint",
                ],
                effort_limit=5.0,
                velocity_limit=1500.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
            ),
        },
    )

    joint_names = (
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    )

    # Original controller
    default_joint_pos = (0.797, -0.168, 0.0, -0.797, 0.168, 0.0)
    p_gains = (20.0, 20.0, 0.0, 20.0, 20.0, 0.0)
    d_gains = (1.0, 1.0, 0.2, 1.0, 1.0, 0.2)
    torque_limits = (50.0, 50.0, 5.0, 50.0, 50.0, 5.0)

    pos_action_scale = 0.8
    vel_action_scale = 10.0
    clip_actions = 100.0

    # Original commands
    command_resampling_time = 5.0
    lin_vel_x_range = (-2.0, 2.0)
    ang_vel_yaw_range = (-2.0, 2.0)
    height_range = (0.18, 0.28)

    # Original observation scaling
    obs_ang_vel_scale = 0.25
    obs_dof_pos_scale = 1.0
    obs_dof_vel_scale = 0.05
    obs_dof_acc_scale = 0.0025
    obs_height_measurements_scale = 5.0
    obs_torque_scale = 0.05
    command_scale = (2.0, 0.25, 5.0)

    # Original observation noise
    add_noise = True
    noise_ang_vel = 0.2 * 0.25
    noise_gravity = 0.05
    noise_dof_pos = 0.02
    noise_dof_vel = 1.5 * 0.05

    obs_history_length = 5

    # Original virtual-leg geometry used by nominal_state reward
    leg_offset = 0.0
    l1 = 0.175
    l2 = 0.208

    fail_to_terminal_time_s = 0.025

    # Original startup controller randomization
    randomize_kp_range = (0.95, 1.05)
    randomize_kd_range = (0.95, 1.05)
    randomize_motor_torque_range = (0.95, 1.05)
    randomize_default_dof_pos_range = (-0.03, 0.03)

    # Original startup physical domain randomization
    friction_range = (0.6, 1.4)
    friction_num_buckets = 64
    restitution_range = (0.6, 1.0)
    added_base_mass_range = (-1.0, 2.0)
    inertia_scale_range = (0.9, 1.1)
    base_com_range = (-0.02, 0.02)

    tracking_sigma = 0.25
    clip_single_reward = 1.0

    # Exact active reward scales from plane/LeggedRobotCfg.
    reward_scales = {
        "tracking_lin_vel": 1.0,
        "tracking_lin_vel_enhance": 1.0,
        "tracking_ang_vel": 1.0,
        "tracking_ang_vel_enhance": 1.0,
        "base_height": 1.0,
        "nominal_state": -1.0,
        "lin_vel_z": -1.0,
        "ang_vel_xy": -0.20,
        "orientation": -100.0,
        "dof_vel": -5.0e-5,
        "dof_acc": -1.0e-7,
        "torques": -1.0e-4,
        "action_rate": -0.01,
        "action_smooth": -0.01,
        "collision": -1.0,
        "dof_pos_limits": -1.0,
    }



