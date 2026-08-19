"""Training-aligned actuator and command constraints for fixed-middle sim2sim."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from controller import EndpointForceController
from run_sim2sim import StudentEvaluator


# Conservative sim2sim/deployment envelope. Training includes bar targets down
# to 0.50 m, but 0.55 m avoids operating at that distribution boundary.
HEIGHT_COMMAND_RANGE = (0.55, 0.85)
VELOCITY_X_COMMAND_RANGE = (-0.5, 0.5)
VELOCITY_Y_COMMAND_RANGE = (-0.5, 0.5)


def smokv3sp_gains_for_joint(name: str) -> tuple[float, float]:
    """Return the exact nominal Kp/Kd values used by fixed-middle training."""

    if "hip_roll" in name:
        return 99.09842777666113, 6.3088018534966395
    if "hip_yaw" in name or "hip_pitch" in name:
        return 40.17923847137318, 2.5578897650279457
    if "knee" in name:
        return 99.09842777666113, 6.3088018534966395
    if "waist_yaw" in name:
        return 40.17923847137318, 2.5578897650279457
    if "waist_roll" in name or "waist_pitch" in name:
        return 28.50124619574858, 1.814445686584846
    if "ankle" in name:
        return 28.50124619574858, 1.814445686584846
    if "shoulder" in name or "elbow" in name:
        return 14.25062309787429, 0.907222843292423
    if "wrist_roll" in name:
        return 14.25062309787429, 0.907222843292423
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return 16.77832748089279, 1.06814150219
    raise KeyError(f"No SMOKV3SP actuator gains are defined for {name!r}")


class TrainingRangeEndpointForceController(EndpointForceController):
    """Endpoint controller that cannot accept out-of-training-range commands."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_targets(
            height=self.requested_height,
            velocity_xy=self.requested_velocity_xy,
        )

    def set_targets(
        self,
        *,
        height: float | None = None,
        velocity_xy: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        bounded_height = None
        if height is not None:
            if not math.isfinite(height):
                raise ValueError("height target must be finite")
            bounded_height = float(np.clip(height, *HEIGHT_COMMAND_RANGE))

        bounded_velocity = None
        if velocity_xy is not None:
            velocity = np.asarray(velocity_xy, dtype=float)
            if velocity.shape != (2,):
                raise ValueError("velocity_xy must contain exactly two values")
            if not np.all(np.isfinite(velocity)):
                raise ValueError("velocity_xy targets must be finite")
            bounded_velocity = np.array(
                [
                    np.clip(velocity[0], *VELOCITY_X_COMMAND_RANGE),
                    np.clip(velocity[1], *VELOCITY_Y_COMMAND_RANGE),
                ],
                dtype=float,
            )

        super().set_targets(
            height=bounded_height,
            velocity_xy=bounded_velocity,
        )


class RobotFrameCommandEndpointForceController(
    TrainingRangeEndpointForceController
):
    """Interpret planar velocity commands in the robot's heading frame.

    The underlying PID, endpoint measurement, force, and MuJoCo application
    remain world-frame quantities. Only the requested horizontal velocity is
    transformed: local +X is robot-forward and local +Y is robot-left. The
    transform is refreshed at every controller evaluation using the current
    floating-base yaw, so roll and pitch do not distort the commanded speed.
    """

    def __init__(self, model: mujoco.MjModel, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.base_qpos_address = int(
            model.joint("floating_base_joint").qposadr[0]
        )
        self.requested_velocity_local_xy = self.requested_velocity_xy.copy()

    @staticmethod
    def _bounded_local_velocity(
        velocity_local_xy: np.ndarray | tuple[float, float],
    ) -> np.ndarray:
        velocity = np.asarray(velocity_local_xy, dtype=float)
        if velocity.shape != (2,):
            raise ValueError("velocity_local_xy must contain exactly two values")
        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_local_xy targets must be finite")
        return np.array(
            [
                np.clip(velocity[0], *VELOCITY_X_COMMAND_RANGE),
                np.clip(velocity[1], *VELOCITY_Y_COMMAND_RANGE),
            ],
            dtype=float,
        )

    def set_local_targets(
        self,
        *,
        height: float | None = None,
        velocity_local_xy: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        if height is not None:
            super().set_targets(height=height)
        if velocity_local_xy is not None:
            self.requested_velocity_local_xy = self._bounded_local_velocity(
                velocity_local_xy
            )

    def _heading_rotation(self, data: mujoco.MjData) -> np.ndarray:
        quaternion = data.qpos[
            self.base_qpos_address + 3 : self.base_qpos_address + 7
        ].astype(float, copy=True)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            raise RuntimeError("Robot root quaternion has zero norm")
        w, x, y, z = quaternion / norm
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return np.array(
            [[cosine, -sine], [sine, cosine]], dtype=float
        )

    def local_to_world_xy(
        self, data: mujoco.MjData, vector_local_xy: np.ndarray
    ) -> np.ndarray:
        return self._heading_rotation(data) @ np.asarray(
            vector_local_xy, dtype=float
        )

    def world_to_local_xy(
        self, data: mujoco.MjData, vector_world_xy: np.ndarray
    ) -> np.ndarray:
        return self._heading_rotation(data).T @ np.asarray(
            vector_world_xy, dtype=float
        )

    def velocity_reference_local_xy(self, data: mujoco.MjData) -> np.ndarray:
        return self.world_to_local_xy(data, self.velocity_reference_xy)

    def _sync_world_velocity_target(self, data: mujoco.MjData) -> None:
        super().set_targets(
            velocity_xy=self.local_to_world_xy(
                data, self.requested_velocity_local_xy
            )
        )

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._sync_world_velocity_target(data)
        super().reset(model, data)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._sync_world_velocity_target(data)
        return super().compute(model, data)


class Smokv3spStudentEvaluator(StudentEvaluator):
    """Student evaluator using the exact nominal actuator gains from training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        gains = [smokv3sp_gains_for_joint(name) for name in self.mujoco_joint_names]
        self.kp = np.array([gain[0] for gain in gains], dtype=np.float64)
        self.kd = np.array([gain[1] for gain in gains], dtype=np.float64)


__all__ = [
    "HEIGHT_COMMAND_RANGE",
    "RobotFrameCommandEndpointForceController",
    "Smokv3spStudentEvaluator",
    "TrainingRangeEndpointForceController",
    "VELOCITY_X_COMMAND_RANGE",
    "VELOCITY_Y_COMMAND_RANGE",
    "smokv3sp_gains_for_joint",
]
