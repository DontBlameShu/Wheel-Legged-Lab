from __future__ import annotations

import argparse
import os
import time
import math

# Preload the working Conda NumPy before Isaac Sim modifies sys.path.
import numpy as np
print("[PLAY] Preloaded NumPy:", np.__version__, np.__file__)

import os
os.environ["WISC_RMUC_PLAY"] = "1"

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play WISC stairs policy.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to model checkpoint.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import torch
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from wheel_legged_robot.tasks.manager_based.wisc_stairs.wisc_stairs_env_cfg import (
    WiscStairsEnvCfg,
)
from wheel_legged_robot.tasks.manager_based.wisc_stairs.wisc_stairs_env import (
    WiscStairsEnv,
)
from wheel_legged_robot.tasks.manager_based.fudan_original.fudan_rsl_rl.modules import (
    ActorCriticSequence,
)


def main():
    checkpoint_path = os.path.abspath(
        os.path.expanduser(args.checkpoint)
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    torch.manual_seed(args.seed)

    # ------------------------------------------------------------
    # Environment: exactly the current WISC stairs environment.
    # ------------------------------------------------------------
    cfg = WiscStairsEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.viewer.origin_type = "world"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (1.35, 0.0, 0.70)
    cfg.viewer.lookat = (0.0, 0.0, 0.20)

    # Play mode only: give more time before automatic reset.
    cfg.episode_length_s = 1.0e9

    # Clean deterministic observation for visual evaluation.
    if hasattr(cfg, "add_noise"):
        cfg.add_noise = False

    env = WiscStairsEnv(cfg=cfg)

    # ------------------------------------------------------------
    # Keyboard command mode.
    #
    # Arrow Up   : +2.50 m/s
    # Arrow Down : -2.50 m/s
    # release    :  0.00 m/s
    #
    # Lateral and yaw commands are disabled because this policy
    # was trained with yaw = 0.
    # ------------------------------------------------------------
    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=2.50,
            v_y_sensitivity=1.50,
            omega_z_sensitivity=0.0,
        )
    )

    # Prevent the environment from replacing keyboard commands
    # with its normal random training commands on reset/resampling.
    env._resample_commands = lambda env_ids: None

    env.commands[:, 0] = 0.0
    env.commands[:, 1] = 0.0
    play_state = {"height": 0.23}
    env._play_power_off = False

    def _toggle_power():
        env._play_power_off = not env._play_power_off
        print("[WISC PLAY] POWER", "OFF" if env._play_power_off else "ON")

    def _height_up():
        play_state["height"] = min(0.30, play_state["height"] + 0.01)
        print(f"[WISC PLAY] HEIGHT = {play_state['height']:.2f} m")

    def _height_down():
        play_state["height"] = max(0.15, play_state["height"] - 0.01)
        print(f"[WISC PLAY] HEIGHT = {play_state['height']:.2f} m")

    keyboard.add_callback("K", _toggle_power)
    keyboard.add_callback("R", _height_up)
    keyboard.add_callback("F", _height_down)

    env.commands[:, 2] = play_state["height"]

    print(keyboard)
    print("[WISC PLAY] KEYBOARD MODE")
    print("  Arrow Up   : forward  2.50 m/s")
    print("  Arrow Down : backward 2.50 m/s")
    print("  Release    : vx = 0")
    print("  yaw        : locked at 0")
    print("  height     : 0.23 m")

    # ------------------------------------------------------------
    # Exact network architecture used by the WISC/Fudan training.
    #
    # actor obs      = 25
    # history        = 5 x 25 = 125
    # latent         = 3
    # actor input    = 25 + 3 = 28
    # critic input   = 141 + 3 = 144
    # actions        = 6
    # ------------------------------------------------------------
    policy = ActorCriticSequence(
        num_obs=25,
        num_critic_obs=144,
        num_actions=6,
        num_encoder_obs=125,
        latent_dim=3,
        encoder_hidden_dims=[128, 64],
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        orthogonal_init=False,
        init_noise_std=0.7,
    ).to(env.device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=env.device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint has no 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    policy.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    policy.eval()

    print("=" * 60)
    print("WISC STAIRS PLAY")
    print("checkpoint :", checkpoint_path)
    print("iteration  :", checkpoint.get("iter", "unknown"))
    print("num_envs   :", args.num_envs)
    print("device     :", env.device)
    print("vx         : +/- 2.00 m/s")
    print("yaw        : 0 rad/s")
    print("=" * 60)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs_history = env.obs_history

    def _update_follow_camera():
        pos = env.robot.data.root_pos_w[0]
        quat = env.robot.data.root_quat_w[0]  # w, x, y, z
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        hx = math.cos(yaw)
        hy = math.sin(yaw)

        # Visual forward is -local-x in this robot model.
        # Camera sits 1.15 m behind it and looks 0.35 m ahead.
        eye = (px + 1.15 * hx, py + 1.15 * hy, pz + 0.55)
        target = (px - 0.35 * hx, py - 0.35 * hy, pz + 0.12)
        env.sim.set_camera_view(eye, target)

    try:
        while simulation_app.is_running():
            step_start = time.perf_counter()
            _update_follow_camera()

            with torch.inference_mode():
                keyboard_cmd = keyboard.advance()
                vx_debug = float(keyboard_cmd[0])
                if not hasattr(env, "_last_vx_debug") or vx_debug != env._last_vx_debug:
                    print(f"[WISC PLAY] keyboard vx = {vx_debug:+.2f} m/s")
                    env._last_vx_debug = vx_debug

                # Robot visual forward is opposite to Se2Keyboard +x.
                env.commands[:, 0] = max(-2.0, min(2.0, -float(keyboard_cmd[0])))
                env.commands[:, 1] = max(-1.5, min(1.5, float(keyboard_cmd[1])))
                env.commands[:, 2] = play_state["height"]

                actions, _ = policy.act_inference(
                    obs,
                    obs_history,
                )

                now_action = time.perf_counter()
                if not hasattr(env, "_last_action_print") or now_action - env._last_action_print >= 0.1:
                    print("[WISC PLAY] action =", [round(float(x), 3) for x in actions[0].detach().cpu()])
                    env._last_action_print = now_action

                (
                    obs_dict,
                    _reward,
                    _terminated,
                    _truncated,
                    _info,
                ) = env.step(actions)

                obs = obs_dict["policy"]
                obs_history = env.obs_history

                now = time.perf_counter()
                if not hasattr(env, "_last_speed_print") or now - env._last_speed_print >= 0.1:
                    actual_vx = float(env.robot.data.root_lin_vel_b[0, 0].item())
                    target_vx = float(env.commands[0, 0].item())
                    qd_dbg = env._dof_vel_from_pos_diff(update_last=False)[0]
                    tq_dbg = env.torques[0]
                    print(f"[WISC PLAY] target={target_vx:+.2f} actual={actual_vx:+.2f}  wheel_qd=({float(qd_dbg[2]):+.2f},{float(qd_dbg[5]):+.2f})  wheel_tau=({float(tq_dbg[2]):+.2f},{float(tq_dbg[5]):+.2f})")
                    env._last_speed_print = now

            # Keep visual playback approximately real-time.
            elapsed = time.perf_counter() - step_start
            sleep_time = env.step_dt - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
