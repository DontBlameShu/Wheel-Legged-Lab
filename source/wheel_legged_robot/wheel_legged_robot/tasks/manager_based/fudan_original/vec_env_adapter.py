from __future__ import annotations

import torch


class FudanVecEnvAdapter:
    """Expose the original fudan_rl_wheel_leg VecEnv interface over DirectRLEnv."""

    def __init__(self, env):
        self.env = env

        self.num_envs = env.num_envs
        self.num_obs = 25
        self.num_privileged_obs = 141
        self.num_actions = 6
        self.obs_history_length = 5

        self._obs = None

    @property
    def device(self):
        return self.env.device

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    def reset(self):
        obs_dict, info = self.env.reset()
        self._obs = obs_dict["policy"]
        return self._obs, info

    def get_observations(self):
        if self._obs is None:
            obs_dict, _ = self.env.reset()
            self._obs = obs_dict["policy"]

        return self._obs, self.env.obs_history

    def get_privileged_observations(self):
        return self.env.get_privileged_observations()

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        self._obs = obs_dict["policy"]

        dones = torch.logical_or(terminated, truncated)

        infos = dict(info)

        # Permanent diagnostic bridge for the original Fudan runner.
        if "log" in infos:
            episode_info = dict(infos.get("episode", {}))
            episode_info.update(infos["log"])
            infos["episode"] = episode_info

        infos["time_outs"] = truncated

        return (
            self._obs,
            self.env.get_privileged_observations(),
            rewards,
            dones,
            infos,
            self.env.obs_history,
        )
