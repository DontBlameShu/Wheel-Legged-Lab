from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils

from .wisc_stairs_env_cfg import WiscStairsEnvCfg


class WiscStairsEnv(DirectRLEnv):
    """Isaac Lab port of the original fudan_rl_wheel_leg plane environment."""

    cfg: WiscStairsEnvCfg

    def __init__(
        self,
        cfg: WiscStairsEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, resolved_names = self.robot.find_joints(
            list(self.cfg.joint_names),
            preserve_order=True,
        )
        if list(resolved_names) != list(self.cfg.joint_names):
            raise RuntimeError(
                f"Joint order mismatch. Expected {self.cfg.joint_names}, "
                f"resolved {resolved_names}."
            )

        self._leg_ids_logical = torch.tensor(
            [0, 1, 3, 4], device=self.device, dtype=torch.long
        )

        n = self.num_envs

        # WISC anti-hopping state.
        # Both wheels must be continuously airborne for >30 ms before penalty.
        self._flight_time = torch.zeros(n, device=self.device)
        self._flight_now = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self._flight_penalty = torch.zeros(
            n, device=self.device
        )

        raw_default = torch.tensor(
            self.cfg.default_joint_pos,
            device=self.device,
            dtype=torch.float,
        )
        self.raw_default_joint_pos = raw_default
        self.default_joint_pos = raw_default.repeat(n, 1)

        # Original startup-only default joint randomization.
        lo, hi = self.cfg.randomize_default_dof_pos_range
        self.default_joint_pos += (
            lo
            + (hi - lo)
            * torch.rand((n, 6), device=self.device)
        )

        p = torch.tensor(
            self.cfg.p_gains, device=self.device, dtype=torch.float
        ).repeat(n, 1)
        d = torch.tensor(
            self.cfg.d_gains, device=self.device, dtype=torch.float
        ).repeat(n, 1)

        lo, hi = self.cfg.randomize_kp_range
        p *= lo + (hi - lo) * torch.rand_like(p)

        lo, hi = self.cfg.randomize_kd_range
        d *= lo + (hi - lo) * torch.rand_like(d)

        self.p_gains = p
        self.d_gains = d

        lo, hi = self.cfg.randomize_motor_torque_range
        self.torque_scale = (
            lo
            + (hi - lo)
            * torch.rand((n, 6), device=self.device)
        )

        self.torque_limits = torch.tensor(
            self.cfg.torque_limits,
            device=self.device,
            dtype=torch.float,
        )

        self.actions = torch.zeros((n, 6), device=self.device)
        self.torques = torch.zeros_like(self.actions)

        # last_actions[:, :, 0] = t-1
        # last_actions[:, :, 1] = t-2
        self.last_actions = torch.zeros((n, 6, 2), device=self.device)

        self.last_joint_vel = torch.zeros((n, 6), device=self.device)
        self.last_base_pos = torch.zeros((n, 3), device=self.device)

        # Original project does not trust the simulator joint velocity for the
        # policy/controller path. It reconstructs qdot from wrapped joint-position
        # differences at every 5 ms physics step.
        self._last_joint_pos_physics = self.robot.data.joint_pos[
            :, self._joint_ids
        ].clone()
        self.dof_vel = torch.zeros((n, 6), device=self.device)

        # Privileged quantities. Physical domain randomization will populate
        # these with randomized values in the next migration stage.
        self.base_mass_deviation = torch.zeros((n, 1), device=self.device)
        self.base_com = torch.zeros((n, 3), device=self.device)
        self.friction_coef = torch.full((n, 1), 0.5, device=self.device)
        self.restitution_coef = torch.full((n, 1), 0.5, device=self.device)
        self._privileged_obs = torch.zeros((n, 141), device=self.device)

        # Match the original Isaac Gym startup domain randomization.
        self._apply_startup_domain_randomization()

        self.commands = torch.zeros((n, 3), device=self.device)
        self.fail_count = torch.zeros(n, device=self.device)
        self._wheel_air_time = torch.zeros((n, 2), device=self.device)
        self._both_wheels_air_time = torch.zeros(n, device=self.device)

        # Consecutive time spent stuck on locally flat terrain.
        self.flat_stuck_count = torch.zeros(n, device=self.device)

        self.obs_history = torch.zeros(
            (n, self.cfg.observation_space * self.cfg.obs_history_length),
            device=self.device,
        )

        self.wrong_direction_count = torch.zeros(
            n, device=self.device
        )

        self._resample_commands(
            torch.arange(n, device=self.device, dtype=torch.long)
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        self.base_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                filter_prim_paths_expr=["/World/ground"],
            )
        )

        self.left_wheel_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/l_wheel_Link",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
            )
        )
        self.right_wheel_contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/r_wheel_Link",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
            )
        )

        # WISC generated stair terrain.
        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain_cfg)

        # Real 11 x 7 terrain height scanner.
        #
        # x: -0.5 ... +0.5  -> 11
        # y: -0.3 ... +0.3  -> 7
        #
        # This exactly preserves the original 77 privileged
        # terrain-height slots.
        self.height_scanner = RayCaster(
            RayCasterCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                update_period=self.step_dt,
                offset=RayCasterCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.50),
                ),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=0.10,
                    size=[1.0, 0.6],
                ),
                debug_vis=False,
                mesh_prim_paths=["/World/ground/terrain/CollisionMesh" if __import__("os").environ.get("WISC_RMUC_PLAY") == "1" else "/World/ground"],
            )
        )

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.terrain.cfg.prim_path])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["base_contact"] = self.base_contact_sensor
        self.scene.sensors["left_wheel_contact"] = self.left_wheel_contact_sensor
        self.scene.sensors["right_wheel_contact"] = self.right_wheel_contact_sensor
        self.scene.sensors["height_scanner"] = self.height_scanner

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:] = torch.clamp(
            actions,
            -self.cfg.clip_actions,
            self.cfg.clip_actions,
        )

    def _apply_action(self) -> None:
        if getattr(self, "_play_power_off", False):
            self.torques.zero_()
            self.robot.set_joint_effort_target(
                self.torques,
                joint_ids=self._joint_ids,
            )
            return

        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self._dof_vel_from_pos_diff(update_last=True)

        pos_ref = self.actions * self.cfg.pos_action_scale
        pos_ref[:, 2] = 0.0
        pos_ref[:, 5] = 0.0

        vel_ref = self.actions * self.cfg.vel_action_scale
        vel_ref[:, 0:2] = 0.0
        vel_ref[:, 3:5] = 0.0
        # Real M3508 + 16.875:1 wheel-speed limit.
        vel_ref[:, 2] = torch.clamp(vel_ref[:, 2], -57.4, 57.4)
        vel_ref[:, 5] = torch.clamp(vel_ref[:, 5], -57.4, 57.4)

        torques = (
            self.p_gains
            * (pos_ref + self.default_joint_pos - q)
            + self.d_gains * (vel_ref - qd)
        )

        torques *= self.torque_scale
        torques = torch.clamp(
            torques,
            -self.torque_limits,
            self.torque_limits,
        )

        self.torques[:] = torques
        self.robot.set_joint_effort_target(
            torques,
            joint_ids=self._joint_ids,
        )

    def _apply_startup_domain_randomization(self) -> None:
        """Faithful startup physics randomization from the original plane task."""
        n = self.num_envs
        env_ids_cpu = torch.arange(n, device="cpu", dtype=torch.long)

        try:
            base_id = list(self.robot.body_names).index("base_link")
        except ValueError as exc:
            raise RuntimeError(
                f"Could not find base_link in bodies: {self.robot.body_names}"
            ) from exc

        # ---------------------------------------------------------------
        # 1. Rigid-shape material.
        #
        # Original:
        #   friction: 64 buckets in [0.6, 1.4], one coefficient per env
        #   restitution: continuous uniform [0.6, 1.0], one value per env
        # All shapes belonging to one robot receive that environment's pair.
        # ---------------------------------------------------------------
        materials = self.robot.root_physx_view.get_material_properties().clone()

        f_lo, f_hi = self.cfg.friction_range
        friction_buckets = torch.empty(
            self.cfg.friction_num_buckets,
            device="cpu",
        ).uniform_(f_lo, f_hi)

        bucket_ids = torch.randint(
            0,
            self.cfg.friction_num_buckets,
            (n,),
            device="cpu",
        )
        friction = friction_buckets[bucket_ids]

        r_lo, r_hi = self.cfg.restitution_range
        restitution = torch.empty(
            n,
            device="cpu",
        ).uniform_(r_lo, r_hi)

        materials[:, :, 0] = friction[:, None]
        materials[:, :, 1] = friction[:, None]
        materials[:, :, 2] = restitution[:, None]

        self.robot.root_physx_view.set_material_properties(
            materials,
            env_ids_cpu,
        )

        self.friction_coef.copy_(
            friction.unsqueeze(-1).to(self.device)
        )
        self.restitution_coef.copy_(
            restitution.unsqueeze(-1).to(self.device)
        )

        # ---------------------------------------------------------------
        # 2. Mass + inertia.
        #
        # Original ordering:
        #   base mass += Uniform[-1, 2] kg
        #   then every rigid body's mass AND inertia are scaled by an
        #   independent Uniform[0.9, 1.1].
        #
        # With recomputeInertia=True in the old Isaac Gym call, inertia
        # follows the final mass ratio. We reproduce that explicitly.
        # ---------------------------------------------------------------
        default_mass = (
            self.robot.data.default_mass.detach().cpu().clone()
        )
        masses = self.robot.root_physx_view.get_masses().clone()

        m_lo, m_hi = self.cfg.added_base_mass_range
        added_mass = torch.empty(
            n,
            device="cpu",
        ).uniform_(m_lo, m_hi)

        masses[:, base_id] = (
            default_mass[:, base_id] + added_mass
        )

        # Important: original privileged observation stores base_mass
        # BEFORE the later global inertia/mass scaling.
        base_mass_for_privileged = masses[:, base_id].clone()

        i_lo, i_hi = self.cfg.inertia_scale_range
        body_scale = torch.empty_like(masses).uniform_(i_lo, i_hi)

        masses *= body_scale
        masses = torch.clamp(masses, min=1.0e-6)

        self.robot.root_physx_view.set_masses(
            masses,
            env_ids_cpu,
        )

        default_inertia = (
            self.robot.data.default_inertia.detach().cpu().clone()
        )
        ratios = masses / default_mass
        inertias = default_inertia * ratios.unsqueeze(-1)

        self.robot.root_physx_view.set_inertias(
            inertias,
            env_ids_cpu,
        )

        centered_base_mass = (
            base_mass_for_privileged
            - base_mass_for_privileged.mean()
        )
        self.base_mass_deviation.copy_(
            centered_base_mass.unsqueeze(-1).to(self.device)
        )

        # ---------------------------------------------------------------
        # 3. Base center of mass.
        # Original independent xyz offsets: +/- 0.02 m.
        # ---------------------------------------------------------------
        coms = self.robot.root_physx_view.get_coms().clone()

        c_lo, c_hi = self.cfg.base_com_range
        com_offset = torch.empty(
            (n, 3),
            device="cpu",
        ).uniform_(c_lo, c_hi)

        coms[:, base_id, :3] += com_offset

        self.robot.root_physx_view.set_coms(
            coms,
            env_ids_cpu,
        )

        self.base_com.copy_(com_offset.to(self.device))

    def _dof_vel_from_pos_diff(
        self,
        update_last: bool = False,
    ) -> torch.Tensor:
        """Original Isaac Gym qdot reconstruction at the 5 ms physics rate."""
        q = self.robot.data.joint_pos[:, self._joint_ids]

        diff = (
            torch.remainder(
                q - self._last_joint_pos_physics + torch.pi,
                2.0 * torch.pi,
            )
            - torch.pi
        )

        vel = diff / self.physics_dt

        if update_last:
            self._last_joint_pos_physics[:] = q
            self.dof_vel[:] = vel

        return vel

    def _base_lin_vel(self) -> torch.Tensor:
        # Original project computes base linear velocity from position difference.
        pos = self.robot.data.root_pos_w
        vel_w = (pos - self.last_base_pos) / self.step_dt
        return math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w,
            vel_w,
        )

    def _base_ang_vel(self) -> torch.Tensor:
        return self.robot.data.root_ang_vel_b

    def _projected_gravity(self) -> torch.Tensor:
        return self.robot.data.projected_gravity_b

    def _virtual_leg_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos[:, self._joint_ids]

        theta1 = torch.stack(
            (q[:, 0], -q[:, 3]),
            dim=1,
        )
        theta2 = torch.stack(
            (
                q[:, 1] + torch.pi / 2.0,
                -q[:, 4] + torch.pi / 2.0,
            ),
            dim=1,
        )

        end_x = (
            self.cfg.leg_offset
            + self.cfg.l1 * torch.cos(theta1)
            + self.cfg.l2 * torch.cos(theta1 + theta2)
        )
        end_y = (
            self.cfg.l1 * torch.sin(theta1)
            + self.cfg.l2 * torch.sin(theta1 + theta2)
        )

        l0 = torch.sqrt(end_x.square() + end_y.square())
        theta0 = torch.atan2(end_y, end_x) - torch.pi / 2.0

        return l0, theta0

    def _build_policy_obs(self, add_noise: bool) -> torch.Tensor:
        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self._dof_vel_from_pos_diff(update_last=False)

        ang_vel = self._base_ang_vel()
        gravity = self._projected_gravity()

        cmd_scale = torch.tensor(
            self.cfg.command_scale,
            device=self.device,
            dtype=torch.float,
        )

        leg_q_rel = (
            q[:, self._leg_ids_logical]
            - self.default_joint_pos[:, self._leg_ids_logical]
        ) * self.cfg.obs_dof_pos_scale

        obs = torch.cat(
            (
                ang_vel * self.cfg.obs_ang_vel_scale,
                gravity,
                self.commands * cmd_scale,
                leg_q_rel,
                qd * self.cfg.obs_dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )

        if add_noise and self.cfg.add_noise:
            noise = torch.zeros_like(obs)

            noise[:, 0:3] = (
                2.0 * torch.rand_like(noise[:, 0:3]) - 1.0
            ) * self.cfg.noise_ang_vel

            noise[:, 3:6] = (
                2.0 * torch.rand_like(noise[:, 3:6]) - 1.0
            ) * self.cfg.noise_gravity

            noise[:, 9:13] = (
                2.0 * torch.rand_like(noise[:, 9:13]) - 1.0
            ) * self.cfg.noise_dof_pos

            noise[:, 13:19] = (
                2.0 * torch.rand_like(noise[:, 13:19]) - 1.0
            ) * self.cfg.noise_dof_vel

            obs = obs + noise

        return torch.clamp(obs, -100.0, 100.0)

    def _terrain_height_samples(self) -> torch.Tensor:
        """Return the real ground z-height at the original 77 scan points."""
        hits = self.height_scanner.data.ray_hits_w[..., 2]

        if hits.shape[-1] != 77:
            raise RuntimeError(
                f"WISC height scanner must produce 77 rays, got {hits.shape}."
            )

        # A ray may temporarily be invalid during initialization/reset.
        # Use the assigned terrain-origin height as a safe fallback.
        fallback = self.terrain.env_origins[:, 2:3].expand_as(hits)

        return torch.where(
            torch.isfinite(hits),
            hits,
            fallback,
        )

    def _base_height_above_ground(self) -> torch.Tensor:
        """Base-link height relative to local terrain directly below it."""
        ground = self._terrain_height_samples()

        # Symmetric 11 x 7 grid -> flattened center ray is index 38.
        center_ground_z = ground[:, 38]

        return (
            self.robot.data.root_pos_w[:, 2]
            - center_ground_z
        )

    def _build_privileged_obs(self) -> torch.Tensor:
        """Original 141D asymmetric-critic observation for the plane task."""
        qd = self._dof_vel_from_pos_diff(update_last=False)

        # Original code uses (last_dof_vel - dof_vel) / control_dt.
        dof_acc = (self.last_joint_vel - qd) / self.step_dt

        policy_obs_clean = self._build_policy_obs(add_noise=False)

        # Same semantic quantity as the original LeggedRobot critic:
        #
        #   root_z - 0.5 - measured_ground_height
        #
        # but measured_ground_height now comes from the real stairs.
        terrain_heights = self._terrain_height_samples()

        heights = (
            torch.clamp(
                self.robot.data.root_pos_w[:, 2:3]
                - 0.5
                - terrain_heights,
                -1.0,
                1.0,
            )
            * self.cfg.obs_height_measurements_scale
        )

        privileged = torch.cat(
            (
                self._base_lin_vel() * 2.0,                     # 3
                policy_obs_clean,                              # 25
                self.last_actions[:, :, 0],                    # 6
                self.last_actions[:, :, 1],                    # 6
                dof_acc * self.cfg.obs_dof_acc_scale,          # 6
                heights,                                       # 77
                self.torques * self.cfg.obs_torque_scale,      # 6
                self.base_mass_deviation,                      # 1
                self.base_com,                                 # 3
                self.default_joint_pos
                - self.raw_default_joint_pos.unsqueeze(0),     # 6
                self.friction_coef,                            # 1
                self.restitution_coef,                         # 1
            ),
            dim=-1,
        )

        if privileged.shape[-1] != 141:
            raise RuntimeError(
                f"Privileged observation must be 141D, got {privileged.shape}."
            )

        return torch.clamp(privileged, -100.0, 100.0)

    def _get_states(self):
        return self._privileged_obs

    def get_privileged_observations(self) -> torch.Tensor:
        return self._privileged_obs

    def get_observation_history(self) -> torch.Tensor:
        return self.obs_history

    def _get_observations(self) -> dict:
        obs = self._build_policy_obs(add_noise=True)

        # Must be built before last_actions/last_joint_vel/last_base_pos advance.
        self._privileged_obs[:] = self._build_privileged_obs()

        self.obs_history[:, :-25] = self.obs_history[:, 25:].clone()
        self.obs_history[:, -25:] = obs

        # These are updated after reward computation, exactly like the old
        # post_physics_step bookkeeping.
        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions

        self.last_joint_vel[:] = self._dof_vel_from_pos_diff(
            update_last=False
        )
        self.last_base_pos[:] = self.robot.data.root_pos_w

        return {"policy": obs}

    def _reward_term(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scaled = (
            value
            * self.cfg.reward_scales[name]
            * self.step_dt
        )

        lim = self.cfg.clip_single_reward * self.step_dt
        if name in ("tracking_lin_vel_enhance", "flight"):
            term = scaled
        else:
            term = torch.clamp(scaled, -lim, lim)

        if not hasattr(self, "_reward_diag_step"):
            self._reward_diag_step = {}

        # Equivalent contribution over one full episode.
        self._reward_diag_step[f"R/{name}"] = (
            term.detach().mean() * float(self.max_episode_length)
        )

        # Especially useful because orientation currently saturates easily.
        if name == "orientation":
            self._reward_diag_step["diag/orientation_clip_frac"] = (
                (scaled.detach().abs() >= (lim - 1.0e-12))
                .float()
                .mean()
            )

        return term

    def _get_rewards(self) -> torch.Tensor:
        q = self.robot.data.joint_pos[:, self._joint_ids]
        qd = self._dof_vel_from_pos_diff(update_last=False)
        base_lin_vel = self._base_lin_vel()
        base_ang_vel = self._base_ang_vel()
        gravity = self._projected_gravity()
        dof_acc = (self.last_joint_vel - qd) / self.step_dt
        _, theta0 = self._virtual_leg_state()

        lin_err = torch.square(self.commands[:, 0] - base_lin_vel[:, 0])
        yaw_err = torch.square(self.commands[:, 1] - base_ang_vel[:, 2])
        base_height = self._base_height_above_ground()
        height_err = torch.square(base_height - self.commands[:, 2])

        rewards = torch.zeros(self.num_envs, device=self.device)

        tracking_lin_vel_reward = self._reward_term(
            "tracking_lin_vel",
            torch.exp(-lin_err / 0.25),
        )
        rewards += tracking_lin_vel_reward
        self._tracking_lin_vel_episode_sum += tracking_lin_vel_reward

        rewards += self._reward_term(
            "tracking_lin_vel_enhance",
            -torch.tanh(lin_err / 0.25),
        )

        tracking_ang_vel_reward = self._reward_term(
            "tracking_ang_vel",
            torch.exp(-yaw_err / 0.25),
        )
        rewards += tracking_ang_vel_reward
        self._tracking_ang_vel_episode_sum += tracking_ang_vel_reward

        rewards += self._reward_term(
            "tracking_ang_vel_enhance",
            torch.exp(-yaw_err / 0.25 / 10.0) - 1.0,
        )

        rewards += self._reward_term(
            "base_height",
            torch.exp(-height_err / 0.001),
        )

        rewards += self._reward_term(
            "nominal_state",
            torch.square(theta0[:, 0]) + torch.square(theta0[:, 1]),
        )

        rewards += self._reward_term(
            "lin_vel_z",
            torch.square(base_lin_vel[:, 2]),
        )

        rewards += self._reward_term(
            "ang_vel_xy",
            torch.sum(torch.square(base_ang_vel[:, :2]), dim=1),
        )

        rewards += self._reward_term(
            "orientation",
            torch.sum(torch.square(gravity[:, :2]), dim=1),
        )

        leg_ids = self._leg_ids_logical
        rewards += self._reward_term(
            "dof_vel",
            torch.sum(torch.square(qd[:, leg_ids]), dim=1),
        )

        rewards += self._reward_term(
            "dof_acc",
            torch.sum(torch.square(dof_acc), dim=1),
        )

        rewards += self._reward_term(
            "torques",
            torch.sum(torch.square(self.torques), dim=1),
        )

        rewards += self._reward_term(
            "action_rate",
            torch.sum(
                torch.square(self.last_actions[:, :, 0] - self.actions),
                dim=1,
            ),
        )

        action_smooth = (
            torch.sum(
                torch.square(
                    self.actions[:, 0:2]
                    - 2.0 * self.last_actions[:, 0:2, 0]
                    + self.last_actions[:, 0:2, 1]
                ),
                dim=1,
            )
            + torch.sum(
                torch.square(
                    self.actions[:, 3:5]
                    - 2.0 * self.last_actions[:, 3:5, 0]
                    + self.last_actions[:, 3:5, 1]
                ),
                dim=1,
            )
        )
        rewards += self._reward_term(
            "action_smooth",
            action_smooth,
        )

        # Fudan WheelLeggedCfg uses penalize_contacts_on = [],
        # so its collision reward is identically zero.
        rewards += self._reward_term(
            "collision",
            torch.zeros(self.num_envs, device=self.device),
        )

        limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]
        q_leg = q[:, leg_ids]
        lo = limits[:, leg_ids, 0]
        hi = limits[:, leg_ids, 1]
        pos_limit_cost = (
            -(q_leg - lo).clamp(max=0.0)
            + (q_leg - hi).clamp(min=0.0)
        ).sum(dim=1)

        rewards += self._reward_term(
            "dof_pos_limits",
            pos_limit_cost,
        )

        # WISC anti-hopping: penalize only sustained BOTH-wheel flight.
        # Contact uses full 3-D force magnitude, so a wheel pushing on a stair
        # riser still counts as supported contact.
        left_force = self.left_wheel_contact_sensor.data.net_forces_w
        right_force = self.right_wheel_contact_sensor.data.net_forces_w
        if hasattr(left_force, "torch"):
            left_force = left_force.torch
        if hasattr(right_force, "torch"):
            right_force = right_force.torch

        left_force_mag = torch.linalg.vector_norm(left_force, dim=-1).amax(dim=1)
        right_force_mag = torch.linalg.vector_norm(right_force, dim=-1).amax(dim=1)
        left_contact = left_force_mag > 1.0
        right_contact = right_force_mag > 1.0
        both_air = (~left_contact) & (~right_contact)
        one_air = left_contact ^ right_contact

        self._flight_now[:] = both_air
        self._flight_time[:] = torch.where(
            both_air,
            self._flight_time + self.step_dt,
            torch.zeros_like(self._flight_time),
        )

        flight_ramp = torch.clamp(
            (self._flight_time - 0.03) / 0.07,
            min=0.0,
            max=1.0,
        )
        self._flight_penalty[:] = torch.where(
            self._flight_time <= 0.10,
            torch.square(flight_ramp),
            torch.clamp(
                1.0 + 10.0 * (self._flight_time - 0.10),
                max=3.0,
            ),
        )
        rewards += self._reward_term(
            "flight",
            self._flight_penalty,
        )

        # Reward diagnostics only -- does not affect training.
        log = dict(getattr(self, "_reward_diag_step", {}))

        self._wheel_air_time[:, 0] = torch.where(
            left_contact,
            torch.zeros_like(self._wheel_air_time[:, 0]),
            self._wheel_air_time[:, 0] + self.step_dt,
        )
        self._wheel_air_time[:, 1] = torch.where(
            right_contact,
            torch.zeros_like(self._wheel_air_time[:, 1]),
            self._wheel_air_time[:, 1] + self.step_dt,
        )
        self._both_wheels_air_time[:] = torch.where(
            both_air,
            self._both_wheels_air_time + self.step_dt,
            torch.zeros_like(self._both_wheels_air_time),
        )

        log.update({
            "diag/wheel_contact_L_frac": left_contact.float().mean(),
            "diag/wheel_contact_R_frac": right_contact.float().mean(),
            "diag/wheel_force_L_mean": left_force_mag.mean(),
            "diag/wheel_force_R_mean": right_force_mag.mean(),
            "diag/one_wheel_air_frac": one_air.float().mean(),
            "diag/both_wheels_air_frac": both_air.float().mean(),
            "diag/wheel_air_L_time_mean": self._wheel_air_time[:, 0].mean(),
            "diag/wheel_air_R_time_mean": self._wheel_air_time[:, 1].mean(),
            "diag/both_air_time_mean": self._both_wheels_air_time.mean(),
            "diag/both_air_time_max": self._both_wheels_air_time.max(),
        })

        pitch_deg = torch.rad2deg(torch.atan2(gravity[:, 0], -gravity[:, 2]))
        roll_deg = torch.rad2deg(torch.atan2(-gravity[:, 1], -gravity[:, 2]))

        log.update(
            {
                "R/total_equiv": rewards.detach().mean() * float(self.max_episode_length),
                "diag/vx_err_abs": torch.sqrt(lin_err).detach().mean(),
                "diag/yaw_err_abs": torch.sqrt(yaw_err).detach().mean(),
                "diag/height_err_abs": torch.sqrt(height_err).detach().mean(),
                "diag/pitch_abs_deg": pitch_deg.detach().abs().mean(),
                "diag/roll_abs_deg": roll_deg.detach().abs().mean(),
                "diag/base_height": base_height.detach().mean(),
                "diag/action_abs": self.actions.detach().abs().mean(),
                "diag/torque_abs": self.torques.detach().abs().mean(),
                "cmd/vx_abs_mean": self.commands[:, 0].detach().abs().mean(),
                "cmd/vx_abs_max": self.commands[:, 0].detach().abs().max(),
                "cmd/yaw_abs_mean": self.commands[:, 1].detach().abs().mean(),
                "cmd/yaw_abs_max": self.commands[:, 1].detach().abs().max(),
                "cmd/height_mean": self.commands[:, 2].detach().mean(),
                "cmd/height_min": self.commands[:, 2].detach().min(),
                "cmd/height_max": self.commands[:, 2].detach().max(),
            }
        )
        self.extras["log"] = log

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Fudan-style command resampling.
        interval = max(
            1,
            int(self.cfg.command_resampling_time / self.step_dt),
        )
        resample_ids = (
            self.episode_length_buf % interval == 0
        ).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Fudan-style fall termination:
        # only a severe fall counts, and it must persist for 1 second.
        fallen = self._projected_gravity()[:, 2] > -0.1
        self.fail_count[:] = torch.where(
            fallen,
            self.fail_count + 1.0,
            torch.zeros_like(self.fail_count),
        )

        fail_steps = self.cfg.fail_to_terminal_time_s / self.step_dt

        # base_link is forbidden from touching the ground.
        base_ground_force = self.base_contact_sensor.data.force_matrix_w
        if hasattr(base_ground_force, "torch"):
            base_ground_force = base_ground_force.torch

        base_ground_contact = (
            torch.linalg.vector_norm(base_ground_force, dim=-1)
            .amax(dim=(1, 2))
            > 1.0
        )

        terminated = self.fail_count > fail_steps

        # Keep only terrain-boundary protection, analogous to Fudan edge reset.
        raw_ground_z = self.height_scanner.data.ray_hits_w[..., 2]
        if hasattr(raw_ground_z, "torch"):
            raw_ground_z = raw_ground_z.torch
        finite_ground = torch.isfinite(raw_ground_z)
        ground_fraction = finite_ground.float().mean(dim=1)
        off_map = (
            (self.episode_length_buf > 5)
            & (ground_fraction < 0.50)
        )
        terminated = terminated | off_map

        time_out = self.episode_length_buf > self.max_episode_length
        return terminated, time_out

    def _ensure_command_curriculum_state(self) -> None:
        """State required by the original Fudan plane command curriculum."""
        if hasattr(self, "_lin_vel_x_command_range"):
            return

        self._lin_vel_x_command_range = torch.tensor(
            self.cfg.lin_vel_x_range,
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        self._tracking_lin_vel_episode_sum = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._tracking_ang_vel_episode_sum = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )

        self._fudan_common_step_counter = 0

    def _update_command_curriculum(self, env_ids: torch.Tensor) -> None:
        """Faithful plane branch of Fudan update_command_curriculum()."""
        if len(env_ids) == 0:
            return

        lin_mean = (
            torch.mean(self._tracking_lin_vel_episode_sum[env_ids])
            / self.max_episode_length
        )
        ang_mean = (
            torch.mean(self._tracking_ang_vel_episode_sum[env_ids])
            / self.max_episode_length
        )

        # Original reward_scales have already been multiplied by dt.
        lin_scale = (
            self.cfg.reward_scales["tracking_lin_vel"]
            * self.step_dt
        )
        ang_scale = (
            self.cfg.reward_scales["tracking_ang_vel"]
            * self.step_dt
        )

        # Original config:
        # curriculum_threshold = 0.7
        # basic_max_curriculum = 4.5
        if (
            lin_mean > 0.7 * lin_scale
            and ang_mean > 0.7 * ang_scale * 0.8
        ):
            self._lin_vel_x_command_range[:, 0] = torch.clamp(
                self._lin_vel_x_command_range[:, 0] - 0.1,
                min=-1.0,
                max=0.0,
            )
            self._lin_vel_x_command_range[:, 1] = torch.clamp(
                self._lin_vel_x_command_range[:, 1] + 0.1,
                min=0.0,
                max=1.0,
            )

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        self._ensure_command_curriculum_state()

        vx_lo = self._lin_vel_x_command_range[env_ids, 0]
        vx_hi = self._lin_vel_x_command_range[env_ids, 1]
        self.commands[env_ids, 0] = (
            vx_lo + (vx_hi - vx_lo) * torch.rand(n, device=self.device)
        )

        yaw_lo, yaw_hi = self.cfg.ang_vel_yaw_range
        self.commands[env_ids, 1] = (
            yaw_lo + (yaw_hi - yaw_lo) * torch.rand(n, device=self.device)
        )

        height_lo, height_hi = self.cfg.height_range
        self.commands[env_ids, 2] = (
            height_lo + (height_hi - height_lo) * torch.rand(n, device=self.device)
        )

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
    ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )

        self._ensure_command_curriculum_state()

        if (
            self._fudan_common_step_counter
            % self.max_episode_length
            == 0
        ):
            self._update_command_curriculum(env_ids)

        self._tracking_lin_vel_episode_sum[env_ids] = 0.0
        self._tracking_ang_vel_episode_sum[env_ids] = 0.0

        super()._reset_idx(env_ids)

        joint_pos = self.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self.robot.data.default_root_state[env_ids].clone()

        # The TerrainImporter origin of an inverted-pyramid staircase is
        # the center of its lowest flat platform.
        root_state[:, :3] += self.terrain.env_origins[env_ids]
        root_state[:, 2] += 0.20
        if __import__("os").environ.get("WISC_RMUC_PLAY") == "1":
            root_state[:, 0] = -1.499993164
            root_state[:, 1] = 1.624343827
            root_state[:, 2] = 0.253535699

        # Uniformly randomize the robot over the pit bottom.
        #
        # Actual middle platform is about 1.8 x 1.8 m for the current
        # 8 m terrain / 1 m border / 0.30 m step configuration.
        # Use the central 1.4 x 1.4 m and leave ~0.20 m edge margin.
        n_reset = env_ids.numel()
        spawn_half_width = 0.0 if __import__("os").environ.get("WISC_RMUC_PLAY") == "1" else 0.70

        root_state[:, 0] += torch.empty(
            n_reset,
            device=self.device,
        ).uniform_(-spawn_half_width, spawn_half_width)

        root_state[:, 1] += torch.empty(
            n_reset,
            device=self.device,
        ).uniform_(-spawn_half_width, spawn_half_width)

        # Start from rest.
        root_state[:, 7:13] = 0.0


        # IMPORTANT:
        # this is the FINAL root-pose write.
        self.robot.write_root_pose_to_sim(
            root_state[:, :7],
            env_ids,
        )
        self.robot.write_root_velocity_to_sim(
            root_state[:, 7:13],
            env_ids,
        )
        self.robot.write_joint_position_to_sim(
            joint_pos,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_joint_velocity_to_sim(
            joint_vel,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )

        self.actions[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.dof_vel[env_ids] = 0.0
        self._last_joint_pos_physics[env_ids] = joint_pos
        self.fail_count[env_ids] = 0.0
        self._wheel_air_time[env_ids] = 0.0
        self._both_wheels_air_time[env_ids] = 0.0
        self.flat_stuck_count[env_ids] = 0.0

        self._flight_time[env_ids] = 0.0
        self._flight_now[env_ids] = False
        self._flight_penalty[env_ids] = 0.0
        self.wrong_direction_count[env_ids] = 0.0

        self.last_base_pos[env_ids] = root_state[:, :3]

        self._resample_commands(env_ids)

        clean_obs = self._build_policy_obs(add_noise=False)
        self.obs_history[env_ids] = clean_obs[env_ids].repeat(
            1,
            self.cfg.obs_history_length,
        )




