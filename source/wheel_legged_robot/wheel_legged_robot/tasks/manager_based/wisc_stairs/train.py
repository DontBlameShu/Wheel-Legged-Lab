from __future__ import annotations

import argparse
import os
import random
import re
from datetime import datetime

# Import conda NumPy before AppLauncher.
# This matches the working Fudan training entry on this machine.
import numpy as np
import torch

from isaaclab.app import AppLauncher


BASELINE_CHECKPOINT = os.path.join(
    "logs",
    "wisc",
    "stairs",
    "plane_baseline_model_8600.pt",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_envs",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--headless",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--log_root",
        type=str,
        default=os.path.join(
            "logs",
            "wisc",
            "stairs",
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume optimizer/iteration state. "
            "Without this flag the checkpoint is loaded as weights only."
        ),
    )

    parser.add_argument("--response_finetune", action="store_true", help="Fine-tune fast command response from an existing checkpoint.")

    return parser.parse_args()


args = parse_args()

launcher = AppLauncher(
    headless=args.headless,
)

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
            "init_noise_std": 0.5,
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
            "max_iterations": args.iterations,
            "save_interval": 100,
            "experiment_name": "wheel_legged",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        },
    }


def checkpoint_iteration(path: str) -> int | None:
    match = re.search(
        r"model_(\d+)\.pt$",
        os.path.basename(path),
    )

    if match is None:
        return None

    return int(match.group(1))


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_cfg = WiscStairsEnvCfg()
    # RESPONSE_FINETUNE_V1: preserve the good policy, train faster command response.
    if args.response_finetune:
        env_cfg.command_resampling_time = 3.0
        env_cfg.reward_scales = dict(env_cfg.reward_scales)
        env_cfg.reward_scales["tracking_lin_vel_enhance"] = 3.0
        env_cfg.reward_scales["action_rate"] = -0.005
        env_cfg.reward_scales["action_smooth"] = -0.005
        print("[WISC RESPONSE FT] command interval = 3.0 s")
        print("[WISC RESPONSE FT] tracking_lin_vel_enhance = 3.0")
        print("[WISC RESPONSE FT] action_rate/action_smooth = -0.005")

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = WiscStairsEnv(
        cfg=env_cfg,
    )

    vec_env = FudanVecEnvAdapter(env)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_dir = os.path.join(
        args.log_root,
        timestamp,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    train_cfg = make_train_cfg()

    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    if args.checkpoint is not None:
        checkpoint_path = os.path.abspath(
            args.checkpoint
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        if args.resume:
            print(
                "[WISC] Resuming full training state:",
                checkpoint_path,
            )

            runner.load(
                checkpoint_path,
                load_optimizer=True,
            )

            recovered = checkpoint_iteration(
                checkpoint_path
            )

            if recovered is not None:
                runner.current_learning_iteration = recovered

        else:
            print(
                "[WISC] Loading PLANE BASELINE WEIGHTS ONLY:",
                checkpoint_path,
            )

            runner.load(
                checkpoint_path,
                load_optimizer=False,
            )

            # WISC training is a new experiment.
            runner.current_learning_iteration = 0

    try:
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
