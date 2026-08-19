"""World-frame bar-yaw control without applying axial bar torque."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class YawTorqueControllerConfig:
    """Physical gains and limits for the bar-vector yaw controller."""

    dt: float = 0.001
    target_yaw_world: float = -0.5 * math.pi
    kp: float = 20.0
    kd: float = 2.0
    torque_limit: float = 5.0
    target_rate_limit: float = math.radians(45.0)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.kp < 0.0 or self.kd < 0.0:
            raise ValueError("yaw gains must be nonnegative")
        if self.torque_limit <= 0.0:
            raise ValueError("torque_limit must be positive")
        if self.target_rate_limit <= 0.0:
            raise ValueError("target_rate_limit must be positive")
        self.target_yaw_world = wrap_angle(self.target_yaw_world)


@dataclass
class YawTorqueSample:
    time: float
    current_vector_world: np.ndarray
    target_vector_world: np.ndarray
    measured_yaw_world: float
    requested_yaw_world: float
    reference_yaw_world: float
    reference_rate: float
    yaw_rate_world: float
    yaw_error: float
    p_torque: float
    d_torque: float
    control_torque_scalar: float
    yaw_axis_world: np.ndarray
    torque_world: np.ndarray
    axial_torque: float
    saturated: bool


class BarYawTorqueController:
    """PD control of the bar-vector's world-frame azimuth.

    The controlled vector follows the training convention: it points from the
    positive-Y bar endpoint (robot side in the original setup) to the
    negative-Y endpoint. Only the vector's XY projection is controlled.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        config: YawTorqueControllerConfig | None = None,
        *,
        body_name: str = "carried_object",
        positive_endpoint_site: str = "positive_y_endpoint",
        negative_endpoint_site: str = "negative_y_endpoint",
        application_site: str = "controller_point",
    ) -> None:
        self.config = config or YawTorqueControllerConfig()
        if self.config.dt < model.opt.timestep:
            raise ValueError(
                "yaw-controller dt must not be faster than the MuJoCo timestep: "
                f"model timestep={model.opt.timestep}, controller dt={self.config.dt}"
            )

        self.body_id = model.body(body_name).id
        self.positive_endpoint_site_id = model.site(positive_endpoint_site).id
        self.negative_endpoint_site_id = model.site(negative_endpoint_site).id
        self.application_site_id = model.site(application_site).id

        self.requested_yaw_world = self.config.target_yaw_world
        self.reference_yaw_world = self.config.target_yaw_world
        self.reference_rate = 0.0
        self._bar_velocity = np.zeros(6, dtype=float)
        self._zero_force = np.zeros(3, dtype=float)

    def set_target_yaw_world(self, yaw: float) -> None:
        self.requested_yaw_world = wrap_angle(yaw)

    def increment_target_yaw_world(self, delta: float) -> None:
        self.set_target_yaw_world(self.requested_yaw_world + float(delta))

    def bar_vector_world(self, data: mujoco.MjData) -> np.ndarray:
        positive = data.site_xpos[self.positive_endpoint_site_id]
        negative = data.site_xpos[self.negative_endpoint_site_id]
        return (negative - positive).copy()

    def measured_yaw_world(self, data: mujoco.MjData) -> float:
        vector = self.bar_vector_world(data)
        horizontal_norm = float(np.linalg.norm(vector[:2]))
        if horizontal_norm < 1e-6:
            raise RuntimeError("bar-vector XY projection is too small to define yaw")
        return math.atan2(float(vector[1]), float(vector[0]))

    def _yaw_rate_world(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> float:
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            self._bar_velocity,
            0,
        )
        vector = self.bar_vector_world(data)
        vector_rate = np.cross(self._bar_velocity[:3], vector)
        horizontal_norm_squared = float(vector[0] ** 2 + vector[1] ** 2)
        if horizontal_norm_squared < 1e-12:
            raise RuntimeError("bar-vector XY projection is too small to define yaw rate")
        return float(
            (vector[0] * vector_rate[1] - vector[1] * vector_rate[0])
            / horizontal_norm_squared
        )

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Start the rate-limited reference from the measured bar yaw."""

        del model
        self.reference_yaw_world = self.measured_yaw_world(data)
        self.reference_rate = 0.0

    def _advance_reference(self) -> None:
        maximum_delta = self.config.target_rate_limit * self.config.dt
        requested_delta = wrap_angle(
            self.requested_yaw_world - self.reference_yaw_world
        )
        applied_delta = float(
            np.clip(requested_delta, -maximum_delta, maximum_delta)
        )
        self.reference_yaw_world = wrap_angle(
            self.reference_yaw_world + applied_delta
        )
        self.reference_rate = applied_delta / self.config.dt

    def compute(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> YawTorqueSample:
        self._advance_reference()
        measured_yaw = self.measured_yaw_world(data)
        yaw_rate_world = self._yaw_rate_world(model, data)
        yaw_error = wrap_angle(self.reference_yaw_world - measured_yaw)
        p_torque = self.config.kp * yaw_error
        d_torque = self.config.kd * (self.reference_rate - yaw_rate_world)
        unsaturated = p_torque + d_torque
        torque_z = float(
            np.clip(
                unsaturated,
                -self.config.torque_limit,
                self.config.torque_limit,
            )
        )

        current_vector = self.bar_vector_world(data)
        unit_vector = current_vector / np.linalg.norm(current_vector)
        world_z = np.array([0.0, 0.0, 1.0], dtype=float)
        # The component of world Z parallel to the bar cannot change its
        # endpoint vector. On a thin bar it produces only enormous axial spin.
        # Remove that null-space component while preserving the complete
        # orientation-changing component of a world-Z yaw torque.
        yaw_axis_world = world_z - float(world_z @ unit_vector) * unit_vector
        torque_world = torque_z * yaw_axis_world
        axial_torque = float(torque_world @ unit_vector)

        target_vector_world = np.array(
            [
                math.cos(self.reference_yaw_world),
                math.sin(self.reference_yaw_world),
                0.0,
            ],
            dtype=float,
        )
        return YawTorqueSample(
            time=float(data.time),
            current_vector_world=current_vector,
            target_vector_world=target_vector_world,
            measured_yaw_world=measured_yaw,
            requested_yaw_world=self.requested_yaw_world,
            reference_yaw_world=self.reference_yaw_world,
            reference_rate=self.reference_rate,
            yaw_rate_world=yaw_rate_world,
            yaw_error=yaw_error,
            p_torque=p_torque,
            d_torque=d_torque,
            control_torque_scalar=torque_z,
            yaw_axis_world=yaw_axis_world,
            torque_world=torque_world,
            axial_torque=axial_torque,
            saturated=not math.isclose(torque_z, unsaturated, abs_tol=1e-12),
        )

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sample: YawTorqueSample,
    ) -> None:
        """Apply the projected world-frame yaw torque with zero net force."""

        mujoco.mj_applyFT(
            model,
            data,
            self._zero_force,
            sample.torque_world,
            data.site_xpos[self.application_site_id],
            self.body_id,
            data.qfrc_applied,
        )
