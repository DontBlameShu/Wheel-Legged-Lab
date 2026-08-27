import argparse
import ctypes
import os
import types
from pathlib import Path

import torch
from pynput import keyboard as pynput_keyboard

from isaaclab.app import AppLauncher


# ============================================================
# Manual control settings
# ============================================================

VX_CMD = 2.0
YAW_CMD = 1.5

HEIGHT_MIN = 0.18
HEIGHT_MAX = 0.28
HEIGHT_DEFAULT = 0.23
HEIGHT_STEP = 0.01

# Space jump macro:
# crouch -> fast extension -> settle
JUMP_CROUCH_S = 0.12
JUMP_EXTEND_S = 0.18
JUMP_SETTLE_S = 0.20


# ============================================================
# 1 / 2 / 3 custom actions
#
# These are command-space macros, so the trained policy still
# generates the actual 6 motor actions.
#
# Change these later however you want.
# ============================================================

CUSTOM_PRESETS = {
    "1": {
        "name": "crouch",
        "vx": 0.0,
        "yaw": 0.0,
        "height": 0.18,
        "duration": 0.80,
    },

    "2": {
        "name": "sprint",
        "vx": 2.5,
        "yaw": 0.0,
        "height": 0.23,
        "duration": 1.20,
    },

    "3": {
        "name": "spin_left",
        "vx": 0.0,
        "yaw": 2.0,
        "height": 0.23,
        "duration": 1.20,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            "logs",
            "fudan_original",
            "wheel_legged",
            "2026-08-24_05-15-22",
            "model_8600.pt",
        ),
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
    )

    return parser.parse_args()


args = parse_args()

launcher = AppLauncher(headless=False)
simulation_app = launcher.app


from wheel_legged_robot.tasks.manager_based.wisc_stairs.wisc_stairs_env_cfg import (
    WiscStairsEnvCfg,
)
from wheel_legged_robot.tasks.manager_based.wisc_stairs.wisc_stairs_env import (
    WiscStairsEnv,
)
from wheel_legged_robot.tasks.manager_based.fudan_original.vec_env_adapter import (
    FudanVecEnvAdapter,
)
from wheel_legged_robot.tasks.manager_based.fudan_original.fudan_rsl_rl.runners import (
    OnPolicyRunner,
)


def make_train_cfg():
    return {
        "policy": {
            "init_noise_std": 0.7,
            "actor_hidden_dims": [128, 64, 32],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
            "num_encoder_obs": 125,
            "latent_dim": 3,
            "encoder_hidden_dims": [128, 64],
        },

        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.005,
            "max_grad_norm": 1.0,
            "extra_learning_rate": 1.0e-3,
        },

        "runner": {
            "policy_class_name": "ActorCriticSequence",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 48,
            "max_iterations": 50000,
            "save_interval": 100,
            "experiment_name": "wheel_legged",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
    }


# ============================================================
# Windows keyboard polling
# No pynput / keyboard package needed.
# ============================================================

class Keyboard:
    def __init__(self):
        self._down = {
            "I": False,
            "K_MOVE": False,
            "J": False,
            "L": False,
            "U": False,
            "O": False,
            "B": False,
            "N": False,
            "7": False,
            "8": False,
            "9": False,
            "ESC": False,
        }

        self._pressed = set()

        self.listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

        self.listener.start()

        print("[INPUT] pynput keyboard enabled")

    def _translate(self, key):
        if key == pynput_keyboard.Key.esc:
            return "ESC"

        try:
            c = key.char.lower()
        except Exception:
            return None

        table = {
            "i": "I",
            "k": "K_MOVE",
            "j": "J",
            "l": "L",
            "u": "U",
            "o": "O",
            "b": "B",
            "n": "N",
            "7": "7",
            "8": "8",
            "9": "9",
        }

        return table.get(c)

    def _on_press(self, key):
        name = self._translate(key)

        if name is None:
            return

        if not self._down[name]:
            self._pressed.add(name)

        self._down[name] = True

    def _on_release(self, key):
        name = self._translate(key)

        if name is None:
            return

        self._down[name] = False

    def down(self, name):
        return self._down.get(name, False)

    def pressed(self, name):
        if name in self._pressed:
            self._pressed.remove(name)
            return True

        return False

    def update(self):
        pass

    def close(self):
        try:
            self.listener.stop()
        except Exception:
            pass


def find_robot(env):
    for name in ("robot", "_robot"):
        obj = getattr(env, name, None)

        if (
            obj is not None
            and hasattr(obj, "set_joint_effort_target")
            and hasattr(obj, "data")
        ):
            return obj

    for obj in vars(env).values():
        if (
            hasattr(obj, "set_joint_effort_target")
            and hasattr(obj, "data")
            and hasattr(obj.data, "joint_pos")
        ):
            return obj

    raise RuntimeError(
        "Could not locate Isaac Lab Articulation in environment."
    )


def set_commands(env, vx, yaw, height):
    vx = max(-2.5, min(2.5, float(vx)))
    yaw = max(-2.0, min(2.0, float(yaw)))
    height = max(
        HEIGHT_MIN,
        min(HEIGHT_MAX, float(height)),
    )

    env.commands[:, 0] = vx
    env.commands[:, 1] = yaw
    env.commands[:, 2] = height


def main():
    checkpoint = Path(
        args.checkpoint
    ).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    env_cfg = WiscStairsEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = WiscStairsEnv(cfg=env_cfg)

    # --------------------------------------------------------
    # Disable random command resampling in PLAY mode.
    # Keyboard owns env.commands instead.
    # --------------------------------------------------------

    if hasattr(env, "_resample_commands"):

        def no_resample_commands(self, env_ids):
            return None

        env._resample_commands = types.MethodType(
            no_resample_commands,
            env,
        )

    vec_env = FudanVecEnvAdapter(env)

    train_cfg = make_train_cfg()

    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=train_cfg,
        log_dir=None,
        device=env.device,
    )

    print(
        f"[PLAY] Loading checkpoint:\n"
        f"{checkpoint}"
    )

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    policy = runner.get_inference_policy(
        device=env.device
    )

    robot = find_robot(env)

    # ========================================================
    # K = actual motor power cut
    #
    # DirectRLEnv calls _apply_action every physics substep.
    # When power is OFF we bypass the controller and write
    # zero effort to every joint.
    # ========================================================

    power = {
        "on": True
    }

    original_apply_action = env._apply_action

    def play_apply_action(self):
        if power["on"]:
            original_apply_action()
        else:
            zero_effort = torch.zeros_like(
                robot.data.joint_pos
            )

            robot.set_joint_effort_target(
                zero_effort
            )

    env._apply_action = types.MethodType(
        play_apply_action,
        env,
    )

    keyboard = Keyboard()

    base_height = HEIGHT_DEFAULT

    jump_step = -1
    jump_saved_height = base_height

    custom = None
    custom_steps_left = 0

    step_dt = float(
        getattr(env, "step_dt", 0.01)
    )

    crouch_steps = max(
        1,
        round(JUMP_CROUCH_S / step_dt),
    )

    extend_steps = max(
        1,
        round(JUMP_EXTEND_S / step_dt),
    )

    settle_steps = max(
        1,
        round(JUMP_SETTLE_S / step_dt),
    )

    jump_total_steps = (
        crouch_steps
        + extend_steps
        + settle_steps
    )

    set_commands(
        env,
        0.0,
        0.0,
        base_height,
    )

    obs, obs_history = (
        vec_env.get_observations()
    )

    print("")
    print("==============================================")
    print(" FUDAN MODEL 8600 - MANUAL PLAY")
    print("==============================================")
    print(" I / K    : forward / backward")
    print(" J / L    : turn left / right")
    print(" U / O    : height up / down")
    print(" B        : MOTOR POWER ON/OFF")
    print(" N        : jump macro")
    print(" 7        : custom action 1 - crouch")
    print(" 8        : custom action 2 - sprint")
    print(" 9        : custom action 3 - spin left")
    print(" ESC      : quit")
    print("==============================================")
    print("")

    i = 0

    try:
        while True:

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if keyboard.down("ESC"):
                print("[PLAY] Exit")
                break

            # ------------------------------------------------
            # MOTOR POWER
            # ------------------------------------------------

            if keyboard.pressed("B"):
                power["on"] = not power["on"]

                if power["on"]:
                    print(
                        "[POWER] ON"
                    )
                else:
                    print(
                        "[POWER] OFF - ZERO MOTOR TORQUE"
                    )

            # ------------------------------------------------
            # HEIGHT
            # U = up, O = down
            height_rate = 0.08

            if (
                keyboard.down("U")
                and not keyboard.down("O")
            ):
                base_height = min(
                    HEIGHT_MAX,
                    base_height + height_rate * step_dt,
                )

            elif (
                keyboard.down("O")
                and not keyboard.down("U")
            ):
                base_height = max(
                    HEIGHT_MIN,
                    base_height - height_rate * step_dt,
                )

            # SPACE = jump macro
            # ------------------------------------------------

            if keyboard.pressed("N"):
                jump_step = 0
                jump_saved_height = base_height

                print(
                    "[JUMP] crouch -> extend"
                )

            # ------------------------------------------------
            # 1 / 2 / 3 custom macros
            # ------------------------------------------------

            for key in ("7", "8", "9"):
                if keyboard.pressed(key):

                    custom = CUSTOM_PRESETS[{"7": "1", "8": "2", "9": "3"}[key]]

                    custom_steps_left = max(
                        1,
                        round(
                            custom["duration"]
                            / step_dt
                        ),
                    )

                    print(
                        f"[CUSTOM {key}] "
                        f"{custom['name']}"
                    )

            # ------------------------------------------------
            # Normal WASD commands
            # ------------------------------------------------

            w = keyboard.down("I")
            s = keyboard.down("K_MOVE")
            a = keyboard.down("J")
            d = keyboard.down("L")

            if w and not s:
                vx = -VX_CMD
            elif s and not w:
                vx = VX_CMD
            else:
                vx = 0.0

            if a and not d:
                yaw = YAW_CMD
            elif d and not a:
                yaw = -YAW_CMD
            else:
                yaw = 0.0

            height = base_height

            # ------------------------------------------------
            # Custom macro overrides WASD temporarily
            # ------------------------------------------------

            if (
                custom is not None
                and custom_steps_left > 0
            ):
                vx = custom["vx"]
                yaw = custom["yaw"]
                height = custom["height"]

                custom_steps_left -= 1

                if custom_steps_left <= 0:
                    print(
                        "[CUSTOM] finished"
                    )
                    custom = None

            # ------------------------------------------------
            # Jump overrides HEIGHT only.
            # WASD remains usable while jumping.
            # ------------------------------------------------

            if jump_step >= 0:

                if jump_step < crouch_steps:
                    height = HEIGHT_MIN

                elif jump_step < (
                    crouch_steps
                    + extend_steps
                ):
                    height = HEIGHT_MAX

                else:
                    height = jump_saved_height

                jump_step += 1

                if jump_step >= jump_total_steps:
                    jump_step = -1
                    base_height = jump_saved_height
                    print(
                        "[JUMP] finished"
                    )

            # ------------------------------------------------
            # Manual commands
            # ------------------------------------------------

            set_commands(
                env,
                vx,
                yaw,
                height,
            )

            # ------------------------------------------------
            # Deterministic policy inference
            # ------------------------------------------------

            with torch.inference_mode():

                policy_output = policy(
                    obs,
                    obs_history,
                )

                if isinstance(
                    policy_output,
                    tuple,
                ):
                    actions = policy_output[0]
                else:
                    actions = policy_output

                (
                    obs,
                    _,
                    _,
                    _,
                    _,
                    obs_history,
                ) = vec_env.step(actions)

            # Restore manual command in case reset touched it.
            set_commands(
                env,
                vx,
                yaw,
                height,
            )

            # ------------------------------------------------
            # Lightweight status display
            # ------------------------------------------------

            if i % 100 == 0:

                try:
                    z = (
                        robot.data.root_pos_w[
                            0, 2
                        ].item()
                    )
                except Exception:
                    z = float("nan")

                print(
                    f"[{i:06d}] "
                    f"power={'ON' if power['on'] else 'OFF'} "
                    f"vx={vx:+.2f} "
                    f"yaw={yaw:+.2f} "
                    f"h={height:.2f} "
                    f"z={z:.3f}"
                )

            keyboard.update()
            i += 1

    finally:
        try:
            keyboard.close()
        except Exception:
            pass

        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

