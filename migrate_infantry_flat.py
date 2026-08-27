from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "source" / "wheel_legged_robot" / "wheel_legged_robot" / "tasks" / "manager_based"
FLAT = BASE / "wheel_legged_robot" / "wheel_legged_flat_env_cfg.py"
ACTIONS = BASE / "wheel_legged_robot" / "mdp" / "actions.py"
DIRECT = BASE / "wheel_legged_robot" / "mdp" / "direct_torque_actions.py"
ASSET = BASE / "wheel_legged_robot" / "assets" / "infantry_v4.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            print(f"[SKIP] {label}: already patched")
            return text
        raise RuntimeError(f"Could not find expected block for {label}")
    return text.replace(old, new, 1)


asset_text = '''from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


INFANTRY_V4_ASSET_DIR = (
    Path(__file__).resolve().parents[2]
    / "fudan_original"
    / "assets"
    / "fudan_infantry_v4"
)


INFANTRY_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(INFANTRY_V4_ASSET_DIR / "urdf" / "infantry_V4_increase.usd"),
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
            solver_velocity_iteration_count=4,
        ),
    ),
    articulation_root_prim_path="/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        # Start above the ground; the flat task will settle to its natural height.
        pos=(0.0, 0.0, 0.32),
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
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
            effort_limit=50.0,
            velocity_limit=1000.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["l_wheel_Joint", "r_wheel_Joint"],
            effort_limit=5.0,
            velocity_limit=57.4,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
    },
)
'''
ASSET.write_text(asset_text, encoding="utf-8")
print("[OK] wrote", ASSET)

# ---------------------------------------------------------------------------
# Make the existing VMC action configurable enough to support both robots.
# Defaults preserve the COD robot behavior; the flat config overrides them for
# infantry_V4.
# ---------------------------------------------------------------------------
actions = ACTIONS.read_text(encoding="utf-8")
actions = replace_once(
    actions,
    '''        # theta1 = [q0, -q3]     (左髋, 右髋)\n        theta1 = torch.stack([-3.1085965751 + dof_pos[:, 0], -3.0737803757 - dof_pos[:, 3]], dim=-1)\n        # theta2 = [q1 + pi/2, -q4 + pi/2]   (左膝, 右膝)\n        theta2 = torch.stack([2.5040614696 - dof_pos[:, 1], 2.4263873329 + dof_pos[:, 4]], dim=-1)''',
    '''        # Robot-specific joint-coordinate -> planar virtual-leg mapping.\n        theta1 = torch.stack(\n            [\n                self.cfg.theta1_offset[0] + self.cfg.theta1_scale[0] * dof_pos[:, 0],\n                self.cfg.theta1_offset[1] + self.cfg.theta1_scale[1] * dof_pos[:, 3],\n            ],\n            dim=-1,\n        )\n        theta2 = torch.stack(\n            [\n                self.cfg.theta2_offset[0] + self.cfg.theta2_scale[0] * dof_pos[:, 1],\n                self.cfg.theta2_offset[1] + self.cfg.theta2_scale[1] * dof_pos[:, 4],\n            ],\n            dim=-1,\n        )''',
    "VMC angle mapping",
)
actions = replace_once(
    actions,
    '''        theta1_dot = torch.stack([dof_vel[:, 0], -dof_vel[:, 3]], dim=-1)\n        theta2_dot = torch.stack([-dof_vel[:, 1], dof_vel[:, 4]], dim=-1)''',
    '''        theta1_dot = torch.stack(\n            [\n                self.cfg.theta1_scale[0] * dof_vel[:, 0],\n                self.cfg.theta1_scale[1] * dof_vel[:, 3],\n            ],\n            dim=-1,\n        )\n        theta2_dot = torch.stack(\n            [\n                self.cfg.theta2_scale[0] * dof_vel[:, 1],\n                self.cfg.theta2_scale[1] * dof_vel[:, 4],\n            ],\n            dim=-1,\n        )''',
    "VMC angle-rate mapping",
)
actions = replace_once(
    actions,
    '''        torques = torch.cat([\n            -T1[:, 0:1],          # left joint 1\n            T2[:, 0:1],           # left joint 2\n            torque_wheel[:, 0:1], # 左轮\n            T1[:, 1:2],           # right joint 1\n            -T2[:, 1:2],          # right joint 2\n            torque_wheel[:, 1:2], # 右轮\n        ], dim=-1)''',
    '''        s = self.cfg.leg_torque_signs\n        torques = torch.cat([\n            s[0] * T1[:, 0:1],\n            s[1] * T2[:, 0:1],\n            torque_wheel[:, 0:1],\n            s[2] * T1[:, 1:2],\n            s[3] * T2[:, 1:2],\n            torque_wheel[:, 1:2],\n        ], dim=-1)''',
    "VMC motor torque mapping",
)
actions = replace_once(
    actions,
    '''    offset: float = 0.054\n    l1: float = 0.15\n    l2: float = 0.25\n\n    # 域随机化开关''',
    '''    offset: float = 0.054\n    l1: float = 0.15\n    l2: float = 0.25\n\n    # Joint-coordinate mapping for virtual-leg kinematics.  These defaults\n    # reproduce the original COD robot exactly.\n    theta1_offset: tuple[float, float] = (-3.1085965751, -3.0737803757)\n    theta1_scale: tuple[float, float] = (1.0, -1.0)\n    theta2_offset: tuple[float, float] = (2.5040614696, 2.4263873329)\n    theta2_scale: tuple[float, float] = (-1.0, 1.0)\n    leg_torque_signs: tuple[float, float, float, float] = (-1.0, 1.0, 1.0, -1.0)\n\n    # 域随机化开关''',
    "VMC configurable mapping fields",
)
ACTIONS.write_text(actions, encoding="utf-8")
print("[OK] patched", ACTIONS)

# Direct-wheel-torque mode uses the same virtual-leg motor sign mapping.
direct = DIRECT.read_text(encoding="utf-8")
direct = replace_once(
    direct,
    '''        torques = torch.cat(\n            [\n                -T1[:, 0:1],\n                T2[:, 0:1],\n                torque_wheel[:, 0:1],\n                T1[:, 1:2],\n                -T2[:, 1:2],\n                torque_wheel[:, 1:2],\n            ],\n            dim=-1,\n        )''',
    '''        s = self.cfg.leg_torque_signs\n        torques = torch.cat(\n            [\n                s[0] * T1[:, 0:1],\n                s[1] * T2[:, 0:1],\n                torque_wheel[:, 0:1],\n                s[2] * T1[:, 1:2],\n                s[3] * T2[:, 1:2],\n                torque_wheel[:, 1:2],\n            ],\n            dim=-1,\n        )''',
    "direct-torque VMC motor mapping",
)
DIRECT.write_text(direct, encoding="utf-8")
print("[OK] patched", DIRECT)

# ---------------------------------------------------------------------------
# Flat task: use infantry_V4 asset and infantry names everywhere.  All later
# original stages inherit from this flat base, so this is the common adapter.
# ---------------------------------------------------------------------------
flat = FLAT.read_text(encoding="utf-8")
flat = replace_once(
    flat,
    'from .assets.wheellegged import WHEEL_LEGGED_CFG',
    'from .assets.wheellegged import WHEEL_LEGGED_CFG\nfrom .assets.infantry_v4 import INFANTRY_V4_CFG',
    "infantry asset import",
)

name_map = {
    "Left_rear_child1_joint": "lf1_Joint",
    "Right_rear_child1_joint": "rf1_Joint",
    "Left_rear_joint": "lf0_Joint",
    "Right_rear_joint": "rf0_Joint",
    "Left_Wheel_joint": "l_wheel_Joint",
    "Right_Wheel_joint": "r_wheel_Joint",
    "Left_rear_child1_link": "lf1_Link",
    "Right_rear_child1_link": "rf1_Link",
    "Left_rear_link": "lf0_Link",
    "Right_rear_link": "rf0_Link",
    "Left_Wheel_link": "l_wheel_Link",
    "Right_Wheel_link": "r_wheel_Link",
}
for old, new in name_map.items():
    flat = flat.replace(old, new)

flat = flat.replace(
    'self.scene.robot = WHEEL_LEGGED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
    'self.scene.robot = INFANTRY_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
)

flat = replace_once(
    flat,
    '''    vmc = VMCActionCfg(\n        asset_name="robot",\n        action_scale_theta=0.35,''',
    '''    vmc = VMCActionCfg(\n        asset_name="robot",\n        joint_names=(\n            "lf0_Joint",\n            "lf1_Joint",\n            "l_wheel_Joint",\n            "rf0_Joint",\n            "rf1_Joint",\n            "r_wheel_Joint",\n        ),\n        action_scale_theta=0.35,''',
    "infantry action joint order",
)
flat = flat.replace('        l0_offset=0.237,', '        l0_offset=0.293,', 1)
flat = flat.replace('        l0_min=0.18,', '        l0_min=0.16,', 1)
flat = flat.replace('        l0_max=0.30,', '        l0_max=0.32,', 1)
flat = flat.replace('        feedforward_force=60.0,', '        feedforward_force=100.0,', 1)
flat = flat.replace('        l1=0.215,', '        l1=0.17472,', 1)
flat = flat.replace('        l2=0.254,', '        l2=0.208,', 1)
flat = replace_once(
    flat,
    '''        l2=0.208,\n    )''',
    '''        l2=0.208,\n        # infantry_V4 planar mapping:\n        # left  theta1=-q0, theta2=-q1-pi/2\n        # right theta1= q0, theta2= q1-pi/2\n        theta1_offset=(0.0, 0.0),\n        theta1_scale=(-1.0, 1.0),\n        theta2_offset=(-math.pi / 2.0, -math.pi / 2.0),\n        theta2_scale=(-1.0, 1.0),\n        # Existing VMC mapping returns the negative generalized torque.\n        # Convert from virtual coordinates back to each physical joint.\n        leg_torque_signs=(1.0, 1.0, -1.0, -1.0),\n    )''',
    "infantry VMC geometry",
)
FLAT.write_text(flat, encoding="utf-8")
print("[OK] patched", FLAT)

print("\nMigration complete. Next run a Flat smoke test before any training.")
