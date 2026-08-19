"""400 Hz Cartesian endpoint controllers for the COLA partner-support model."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np


DEFAULT_HEIGHT_RANGE = (0.4, 0.9)
DEFAULT_VELOCITY_X_RANGE = (-0.5, 0.5)
DEFAULT_VELOCITY_Y_RANGE = (-0.5, 0.5)


def range_mean(bounds: tuple[float, float]) -> float:
    return 0.5 * (bounds[0] + bounds[1])


def clamp_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """Return a copy whose Euclidean norm is no larger than ``limit``."""

    magnitude = float(np.linalg.norm(vector))
    if magnitude <= limit or magnitude == 0.0:
        return vector.copy()
    return vector * (limit / magnitude)


@dataclass
class EndpointControllerConfig:
    """Controller gains in physical units, evaluated once per physics step."""

    dt: float = 0.0025

    target_height: float = range_mean(DEFAULT_HEIGHT_RANGE)
    target_velocity_xy: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                range_mean(DEFAULT_VELOCITY_X_RANGE),
                range_mean(DEFAULT_VELOCITY_Y_RANGE),
            ],
            dtype=float,
        )
    )

    # Vertical endpoint position PD: N/m and N*s/m.
    height_kp: float = 800.0
    height_kd: float = 35.0
    height_force_limit: float = 300.0
    height_target_rate_limit: float = 0.20

    # Horizontal endpoint velocity PID: N/(m/s), N/m, and N/(m/s^2).
    velocity_kp: float = 30.0
    velocity_ki: float = 60.0
    velocity_kd: float = 0.1
    horizontal_force_limit: float = 100.0
    integral_force_limit: float = 15.0
    derivative_cutoff_hz: float = 20.0
    velocity_target_slew_limit: float = 2.0
    velocity_error_deadband: float = 0.005

    # None means use the carried body's fixed mass times world-Z gravity.  This
    # matches the Isaac training controller's ``bar_mass * 9.81`` term.
    gravity_feedforward: float | None = None

    def __post_init__(self) -> None:
        self.target_velocity_xy = np.asarray(
            self.target_velocity_xy, dtype=float
        ).copy()
        if self.target_velocity_xy.shape != (2,):
            raise ValueError("target_velocity_xy must contain exactly two values")

        nonnegative = {
            "height_kp": self.height_kp,
            "height_kd": self.height_kd,
            "velocity_kp": self.velocity_kp,
            "velocity_ki": self.velocity_ki,
            "velocity_kd": self.velocity_kd,
            "integral_force_limit": self.integral_force_limit,
            "velocity_error_deadband": self.velocity_error_deadband,
        }
        positive = {
            "dt": self.dt,
            "height_force_limit": self.height_force_limit,
            "height_target_rate_limit": self.height_target_rate_limit,
            "horizontal_force_limit": self.horizontal_force_limit,
            "derivative_cutoff_hz": self.derivative_cutoff_hz,
            "velocity_target_slew_limit": self.velocity_target_slew_limit,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ControllerSample:
    time: float
    endpoint_position: np.ndarray
    endpoint_velocity: np.ndarray
    height_reference: float
    velocity_reference_xy: np.ndarray
    height_error: float
    velocity_error_xy: np.ndarray
    force_world: np.ndarray
    height_p_force: float
    height_d_force: float
    height_gravity_force: float
    velocity_p_force_xy: np.ndarray
    velocity_i_force_xy: np.ndarray
    velocity_d_force_xy: np.ndarray


class EndpointForceController:
    """PD height and PID horizontal-velocity control at one MuJoCo site.

    The returned force is expressed in the world frame and is applied at the
    configured site. No controller torque is supplied.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        config: EndpointControllerConfig | None = None,
        *,
        site_name: str = "human_endpoint",
        body_name: str = "carried_object",
    ) -> None:
        self.config = config or EndpointControllerConfig()
        if self.config.dt < model.opt.timestep:
            raise ValueError(
                "controller dt must not be faster than the MuJoCo timestep: "
                f"model timestep={model.opt.timestep}, controller dt={self.config.dt}"
            )

        self.site_id = model.site(site_name).id
        self.body_id = model.body(body_name).id

        self.requested_height = float(self.config.target_height)
        self.requested_velocity_xy = np.asarray(
            self.config.target_velocity_xy, dtype=float
        ).copy()

        self.height_reference = self.requested_height
        self.height_reference_velocity = 0.0
        self.velocity_reference_xy = self.requested_velocity_xy.copy()
        self.velocity_integral_xy = np.zeros(2, dtype=float)
        self.filtered_acceleration_xy = np.zeros(2, dtype=float)
        self.previous_velocity_xy = np.zeros(2, dtype=float)
        self.gravity_feedforward = 0.0

        self._jacobian_position = np.zeros((3, model.nv), dtype=float)
        self._jacobian_rotation = np.zeros((3, model.nv), dtype=float)
        self._zero_torque = np.zeros(3, dtype=float)

    def set_targets(
        self,
        *,
        height: float | None = None,
        velocity_xy: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        """Set requested targets; internal references approach them with rate limits."""

        if height is not None:
            self.requested_height = float(height)
        if velocity_xy is not None:
            velocity_array = np.asarray(velocity_xy, dtype=float)
            if velocity_array.shape != (2,):
                raise ValueError("velocity_xy must contain exactly two values")
            self.requested_velocity_xy = velocity_array.copy()

    def endpoint_state(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return endpoint position and linear velocity in the world frame."""

        mujoco.mj_jacSite(
            model,
            data,
            self._jacobian_position,
            self._jacobian_rotation,
            self.site_id,
        )
        position = data.site_xpos[self.site_id].copy()
        velocity = self._jacobian_position @ data.qvel
        return position, velocity

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Clear dynamic controller state after an environment reset."""

        position, velocity = self.endpoint_state(model, data)
        self.height_reference = float(position[2])
        self.height_reference_velocity = 0.0
        self.velocity_reference_xy = velocity[:2].copy()
        self.velocity_integral_xy.fill(0.0)
        self.filtered_acceleration_xy.fill(0.0)
        self.previous_velocity_xy = velocity[:2].copy()
        self.gravity_feedforward = self._vertical_feedforward(model, data)

    def _vertical_feedforward(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> float:
        if self.config.gravity_feedforward is not None:
            return float(self.config.gravity_feedforward)
        # Use only the carried body's declared mass, as in Isaac training.
        # Equality-constraint reactions and the robot's apparent mass must not
        # alter this controller term.
        del data
        return float(model.body_mass[self.body_id] * -model.opt.gravity[2])

    @staticmethod
    def _rate_limit(current: np.ndarray, requested: np.ndarray, max_delta: float) -> np.ndarray:
        return current + np.clip(requested - current, -max_delta, max_delta)

    def _advance_references(self) -> None:
        cfg = self.config
        height_delta = np.clip(
            self.requested_height - self.height_reference,
            -cfg.height_target_rate_limit * cfg.dt,
            cfg.height_target_rate_limit * cfg.dt,
        )
        self.height_reference += float(height_delta)
        self.height_reference_velocity = float(height_delta / cfg.dt)
        self.velocity_reference_xy = self._rate_limit(
            self.velocity_reference_xy,
            self.requested_velocity_xy,
            cfg.velocity_target_slew_limit * cfg.dt,
        )

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> ControllerSample:
        """Evaluate the two controllers without mutating MuJoCo force buffers."""

        cfg = self.config
        self._advance_references()
        position, velocity = self.endpoint_state(model, data)
        # Vertical position PD plus static gravity feedforward.
        height_error = self.height_reference - position[2]
        height_p = cfg.height_kp * height_error
        height_d = cfg.height_kd * (self.height_reference_velocity - velocity[2])
        height_force = np.clip(
            height_p + height_d + self.gravity_feedforward,
            -cfg.height_force_limit,
            cfg.height_force_limit,
        )

        # Horizontal velocity PID. The D term differentiates the measurement,
        # avoiding a derivative kick when the velocity request changes.
        raw_acceleration_xy = (velocity[:2] - self.previous_velocity_xy) / cfg.dt
        alpha = math.exp(-2.0 * math.pi * cfg.derivative_cutoff_hz * cfg.dt)
        self.filtered_acceleration_xy = (
            alpha * self.filtered_acceleration_xy
            + (1.0 - alpha) * raw_acceleration_xy
        )
        self.previous_velocity_xy = velocity[:2].copy()

        velocity_error = self.velocity_reference_xy - velocity[:2]
        integration_error = velocity_error.copy()
        integration_error[np.abs(integration_error) < cfg.velocity_error_deadband] = 0.0

        p_force = cfg.velocity_kp * velocity_error
        d_force = -cfg.velocity_kd * self.filtered_acceleration_xy
        current_i_force = cfg.velocity_ki * self.velocity_integral_xy
        unsaturated_force = p_force + current_i_force + d_force

        # Conditional integration anti-windup: freeze only when saturated and
        # the current error would push farther into saturation.
        unsaturated_magnitude = float(np.linalg.norm(unsaturated_force))
        if unsaturated_magnitude >= cfg.horizontal_force_limit:
            force_direction = unsaturated_force / unsaturated_magnitude
            outward_error = float(integration_error @ force_direction)
            if outward_error > 0.0:
                # Remove only the error component that pushes farther outward;
                # tangential and inward integration remain active.
                integration_error -= outward_error * force_direction
        if cfg.velocity_ki > 0.0:
            self.velocity_integral_xy += integration_error * cfg.dt
            integral_force = clamp_vector_norm(
                cfg.velocity_ki * self.velocity_integral_xy,
                cfg.integral_force_limit,
            )
            self.velocity_integral_xy = integral_force / cfg.velocity_ki
            i_force = integral_force
        else:
            self.velocity_integral_xy.fill(0.0)
            i_force = np.zeros(2, dtype=float)
        horizontal_force = clamp_vector_norm(
            p_force + i_force + d_force,
            cfg.horizontal_force_limit,
        )

        force_world = np.array(
            [horizontal_force[0], horizontal_force[1], height_force], dtype=float
        )
        return ControllerSample(
            time=float(data.time),
            endpoint_position=position,
            endpoint_velocity=velocity,
            height_reference=float(self.height_reference),
            velocity_reference_xy=self.velocity_reference_xy.copy(),
            height_error=float(height_error),
            velocity_error_xy=velocity_error.copy(),
            force_world=force_world,
            height_p_force=float(height_p),
            height_d_force=float(height_d),
            height_gravity_force=float(self.gravity_feedforward),
            velocity_p_force_xy=p_force.copy(),
            velocity_i_force_xy=i_force.copy(),
            velocity_d_force_xy=d_force.copy(),
        )

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sample: ControllerSample,
    ) -> None:
        """Apply only the sampled world-frame force at the configured site."""

        mujoco.mj_applyFT(
            model,
            data,
            sample.force_world,
            self._zero_torque,
            sample.endpoint_position,
            self.body_id,
            data.qfrc_applied,
        )


def load_model(xml_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the single-object model and reset it to its home keyframe."""

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data
