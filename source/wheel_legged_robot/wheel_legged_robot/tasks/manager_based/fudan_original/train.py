from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--log_root",
        type=str,
        default=os.path.join("logs", "fudan_original", "wheel_legged"),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


args = parse_args()

launcher = AppLauncher(headless=args.headless)
simulation_app = launcher.app

from wheel_legged_robot.tasks.manager_based.fudan_original.fudan_env_cfg import FudanPlaneEnvCfg
from wheel_legged_robot.tasks.manager_based.fudan_original.fudan_env import FudanPlaneEnv
from wheel_legged_robot.tasks.manager_based.fudan_original.vec_env_adapter import FudanVecEnvAdapter
from wheel_legged_robot.tasks.manager_based.fudan_original.fudan_rsl_rl.runners import OnPolicyRunner


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


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env_cfg = FudanPlaneEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = FudanPlaneEnv(cfg=env_cfg)
    vec_env = FudanVecEnvAdapter(env)
    train_cfg = make_train_cfg()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(args.log_root, timestamp))
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print("FUDAN ORIGINAL PLANE TRAINING")
    print("num_envs       :", args.num_envs)
    print("iterations     :", args.iterations)
    print("steps/env/iter :", train_cfg["runner"]["num_steps_per_env"])
    print("policy obs     :", vec_env.num_obs)
    print("history obs    :", vec_env.obs_history_length * vec_env.num_obs)
    print("privileged obs :", vec_env.num_privileged_obs)
    print("critic input   :", vec_env.num_privileged_obs + train_cfg["policy"]["latent_dim"])
    print("log_dir        :", log_dir)
    print("=" * 80)

    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    if args.checkpoint is not None:
        checkpoint_path = os.path.abspath(args.checkpoint)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        runner.load(checkpoint_path)

        # Trust model_XXX.pt filename for old Fudan checkpoints.
        checkpoint_stem = os.path.splitext(
            os.path.basename(checkpoint_path)
        )[0]

        try:
            runner.current_learning_iteration = int(
                checkpoint_stem.rsplit("_", 1)[1]
            )
        except (IndexError, ValueError):
            pass

        print(
            f"[INFO] Resume iteration: "
            f"{runner.current_learning_iteration}"
        )

    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=True,
    )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
