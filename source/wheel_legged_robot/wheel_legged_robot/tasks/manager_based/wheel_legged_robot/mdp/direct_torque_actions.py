# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

from .actions import VMCAction, VMCActionCfg as VelocityReferenceVMCActionCfg


class DirectTorqueVMCAction(VMCAction):
    """VMC leg control with direct wheel-torque policy actions.

    Action order remains six-dimensional so the observation/action interface is
    unchanged, but wheel action semantics now match the real robot:

        [theta0_ref_L, L0_ref_L, wheel_torque_L,
         theta0_ref_R, L0_ref_R, wheel_torque_R]

    The leg channels still generate motor torques through the VMC Jacobian. The
    wheel channels bypass the old simulated wheel-speed PI loop completely.
    """

    cfg: DirectTorqueVMCActionCfg

    def __init__(self, cfg: DirectTorqueVMCActionCfg, env):
        super().__init__(cfg, env)
        self.debug_wheel_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self.debug_wheel_torque = torch.zeros_like(self.debug_wheel_vel)
        # Kept only so existing debug logging does not break. In direct-torque
        # mode this is the requested wheel torque, not a velocity reference.
        self.debug_wheel_ref = torch.zeros_like(self.debug_wheel_vel)
        self.debug_wheel_integral = torch.zeros_like(self.debug_wheel_vel)

    def apply_actions(self):
        # Refresh virtual-leg kinematics at the 1 kHz physics rate.
        dof_pos, dof_vel, theta1, theta2, L0, theta0, theta0_dot, L0_dot = (
            self._compute_kinematics()
        )

        actions = self._processed_actions

        # Leg task-space references.
        theta0_ref = torch.stack(
            [
                actions[:, 0] * self.cfg.action_scale_theta,
                actions[:, 3] * self.cfg.action_scale_theta,
            ],
            dim=-1,
        )
        L0_ref = torch.stack(
            [
                torch.clamp(
                    actions[:, 1] * self.cfg.action_scale_l0 + self.cfg.l0_offset,
                    self.cfg.l0_min,
                    self.cfg.l0_max,
                ),
                torch.clamp(
                    actions[:, 4] * self.cfg.action_scale_l0 + self.cfg.l0_offset,
                    self.cfg.l0_min,
                    self.cfg.l0_max,
                ),
            ],
            dim=-1,
        )

        # Wheel channels are direct output-shaft torque commands. No velocity
        # target, PI controller, or integral state is present in the control
        # path. This matches the real robot's torque-command interface.
        torque_wheel = torch.stack([actions[:, 2], actions[:, 5]], dim=-1)
        torque_wheel = torque_wheel * self.cfg.action_scale_torque

        self.debug_wheel_vel.copy_(dof_vel[:, [2, 5]].detach())
        self.debug_wheel_torque.copy_(torque_wheel.detach())
        self.debug_wheel_ref.copy_(torque_wheel.detach())
        self.debug_wheel_integral.zero_()

        # Leg VMC task-space control.
        theta0_error = torch.atan2(torch.sin(theta0_ref - theta0), torch.cos(theta0_ref - theta0))
        torque_theta = self._theta_kp * theta0_error - self._theta_kd * theta0_dot
        l0_kp, l0_kd, feedforward_force = self._get_l0_control_parameters()
        force_L0 = l0_kp * (L0_ref - L0) - l0_kd * L0_dot
        force_L0 += feedforward_force

        # Map virtual-leg force/torque to the four physical leg motors.
        T1, T2 = self._vmc_mapping(theta1, theta2, L0, theta0, force_L0, torque_theta)

        torques = torch.cat(
            [
                -T1[:, 0:1],
                T2[:, 0:1],
                torque_wheel[:, 0:1],
                T1[:, 1:2],
                -T2[:, 1:2],
                torque_wheel[:, 1:2],
            ],
            dim=-1,
        )

        # Preserve actuator-strength domain randomization and power-on/off gate.
        torques = torques * self._torque_scale
        torques = torques * self._motor_enable_scale.unsqueeze(-1)

        # Final clamp is per physical motor. With the hardware actuator config
        # this is +/-25 N m for each leg motor and +/-4 N m for each wheel.
        effort_limits = self._asset.data.joint_effort_limits[:, self._joint_ids]
        self._torque_saturation[:] = torch.mean(
            (torch.abs(torques) > effort_limits).float(), dim=1
        )
        torques = torch.clamp(torques, -effort_limits, effort_limits)

        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)


@configclass
class DirectTorqueVMCActionCfg(VelocityReferenceVMCActionCfg):
    """Direct-wheel-torque variant of the existing VMC action configuration."""

    class_type: type[ActionTerm] = DirectTorqueVMCAction

    # M3508 + custom 268/17 reduction, direct output-shaft torque command.
    # +/-1 policy action corresponds to the selected short-duration peak limit.
    action_scale_torque: float = 4.0
