#!/usr/bin/env python3
"""Local web UI for the COLA fixed-middle-bar MuJoCo sim2sim pipeline.

The browser is deliberately only a control and display surface. Native MuJoCo,
the exported TorchScript student, the 29-DOF robot PD, and all three external
bar controllers run from the website-local sim2sim runtime copied under
``demo/sim2sim_runtime``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Callable

# MuJoCo must select its headless renderer before it is imported by the shared
# sim2sim modules. The user can override this for a different local platform.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw


DEMO_ROOT = Path(__file__).resolve().parent
SITE_ROOT = DEMO_ROOT.parent
SIM2SIM_ROOT = (
    DEMO_ROOT
    / "sim2sim_runtime"
    / "mujoco_fixed_bar_middle_controller_sim2sim"
)
DEFAULT_POLICY = (
    DEMO_ROOT
    / "models"
    / "policy_student_loop_static_three_jitter_phase3_model_5000.jit"
)

if not SIM2SIM_ROOT.is_dir():
    raise RuntimeError(f"Fixed-middle sim2sim pipeline not found: {SIM2SIM_ROOT}")
sys.path.insert(0, str(SIM2SIM_ROOT))

import run_fixed_bar_middle as sim  # noqa: E402


PHYSICS_FREQUENCY = 1000.0
CONTROLLER_FREQUENCY = sim.CONTROLLER_FREQUENCY
POLICY_FREQUENCY = sim.POLICY_FREQUENCY
DEFAULT_HEIGHT = 0.70
DEFAULT_YAW_DEG = -90.0
JPEG_BOUNDARY = b"cola-frame"
HEIGHT_ARROW_COLOR = np.array([0.10, 0.45, 1.0, 0.95], dtype=np.float32)
VELOCITY_ARROW_COLOR = np.array([1.0, 0.15, 0.80, 0.95], dtype=np.float32)


@dataclass
class CameraState:
    azimuth: float = 138.0
    elevation: float = -13.0
    distance: float = 3.35
    follow: bool = True


class ColaSimulation:
    """Thread-safe native simulation shared by HTTP clients."""

    def __init__(
        self,
        policy_path: Path,
        *,
        device: str = "cpu",
        render_width: int = 3840,
        render_height: int = 2160,
        render_fps: float = 30.0,
        controllers_enabled: bool = True,
        enable_rendering: bool = True,
    ) -> None:
        self.policy_path = policy_path.resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Student policy not found: {self.policy_path}")

        self.args = sim.parser().parse_args(
            [
                "--policy",
                str(self.policy_path),
                "--device",
                device,
                "--simulation-frequency",
                str(PHYSICS_FREQUENCY),
                "--height",
                str(DEFAULT_HEIGHT),
                "--target-yaw-deg",
                str(DEFAULT_YAW_DEG),
            ]
        )
        self.args.enable_bar_controllers = controllers_enabled

        self.model, self.data = sim.load_model(str(self.args.model.resolve()))
        self.model.opt.timestep = 1.0 / PHYSICS_FREQUENCY
        ground_id = self.model.geom("ground").id
        if self.model.geom_type[ground_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise RuntimeError("The named ground geom must be a MuJoCo plane")
        self.model.geom_size[ground_id, :2] = 0.0
        self.policy_decimation = int(round(PHYSICS_FREQUENCY / POLICY_FREQUENCY))
        if not math.isclose(
            PHYSICS_FREQUENCY / POLICY_FREQUENCY,
            self.policy_decimation,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Physics frequency must be an integer multiple of policy frequency")

        self.evaluator = sim.StudentEvaluator(
            self.model, self.data, self.policy_path, self.args.device
        )
        self.endpoint_controller = sim.EndpointForceController(
            self.model,
            sim.make_config(self.args, dt=1.0 / CONTROLLER_FREQUENCY),
            site_name="controller_point",
        )
        self.endpoint_controller.set_local_targets(
            height=self.args.height,
            velocity_local_xy=(self.args.vx, self.args.vy),
        )
        self.endpoint_controller.reset(self.model, self.data)
        self.yaw_controller = sim.BarYawTorqueController(
            self.model,
            sim.YawTorqueControllerConfig(
                dt=1.0 / CONTROLLER_FREQUENCY,
                target_yaw_world=math.radians(self.args.target_yaw_deg),
                kp=self.args.yaw_kp,
                kd=self.args.yaw_kd,
                torque_limit=self.args.yaw_torque_limit,
                target_rate_limit=math.radians(self.args.yaw_target_rate_deg_s),
            ),
        )
        self.yaw_controller.set_target_yaw_world(
            math.radians(self.args.target_yaw_deg)
        )
        self.yaw_controller.reset(self.model, self.data)

        self.lock = threading.RLock()
        self.frame_condition = threading.Condition()
        self.latest_frame = self._placeholder_frame("Starting native MuJoCo…")
        self.frame_sequence = 0
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), self.render_width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), self.render_height
        )
        self.render_period = 1.0 / float(render_fps)
        self.enable_rendering = bool(enable_rendering)
        self.camera = CameraState()

        self.physics_step = 0
        self.controller_phase = 0.0
        self.endpoint_sample = None
        self.yaw_sample = None
        self.controllers_enabled = bool(controllers_enabled)
        self.controller_overlays = {
            "velocity": False,
            "height": False,
            "torque": False,
        }
        self.paused = False
        self.stop_requested = False
        self.runtime_error: str | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _placeholder_frame(message: str) -> bytes:
        image = Image.new("RGB", (1280, 720), "#071018")
        draw = ImageDraw.Draw(image)
        draw.text((56, 62), "COLA / MuJoCo", fill="#f3f8fb")
        draw.text((56, 104), message, fill="#80d7ff")
        payload = io.BytesIO()
        image.save(payload, format="JPEG", quality=86)
        return payload.getvalue()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="cola-mujoco", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.stop_requested = True
        if self._thread is not None:
            self._thread.join(timeout=4.0)

    def _reset_unlocked(self) -> None:
        self.evaluator.reset_simulation()
        self.endpoint_controller.reset(self.model, self.data)
        self.yaw_controller.reset(self.model, self.data)
        self.physics_step = 0
        self.controller_phase = 0.0
        self.endpoint_sample = None
        self.yaw_sample = None
        self.runtime_error = None

    def _step_unlocked(self) -> None:
        if self.physics_step % self.policy_decimation == 0:
            self.evaluator.infer_action()

        self.data.qfrc_applied.fill(0.0)
        self.evaluator.apply_robot_pd()

        controller_due = self.physics_step == 0
        if not controller_due:
            self.controller_phase += CONTROLLER_FREQUENCY
            if self.controller_phase >= PHYSICS_FREQUENCY:
                self.controller_phase -= PHYSICS_FREQUENCY
                controller_due = True

        if controller_due:
            self.endpoint_sample = self.endpoint_controller.compute(
                self.model, self.data
            )
            self.yaw_sample = self.yaw_controller.compute(self.model, self.data)

        if self.endpoint_sample is None or self.yaw_sample is None:
            raise RuntimeError("Controller samples were not initialized")
        if self.controllers_enabled:
            self.endpoint_controller.apply(
                self.model, self.data, self.endpoint_sample
            )
            self.yaw_controller.apply(self.model, self.data, self.yaw_sample)

        mujoco.mj_step(self.model, self.data)
        self.physics_step += 1

    def run_steps_for_test(self, count: int) -> dict[str, Any]:
        with self.lock:
            for _ in range(count):
                self._step_unlocked()
            return self._state_unlocked()

    def _render_unlocked(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera) -> bytes:
        if self.camera.follow:
            base = self.evaluator.base_position
            camera.lookat[:] = [base[0] + 0.10, base[1], 0.78]
        else:
            camera.lookat[:] = [0.10, 0.0, 0.78]
        camera.azimuth = self.camera.azimuth
        camera.elevation = self.camera.elevation
        camera.distance = self.camera.distance

        renderer.update_scene(self.data, camera=camera)
        if self.endpoint_sample is not None and self.yaw_sample is not None:
            if self.controllers_enabled:
                vertical_force = np.array(
                    [0.0, 0.0, self.endpoint_sample.force_world[2]]
                )
                horizontal_force = np.array(
                    [
                        self.endpoint_sample.force_world[0],
                        self.endpoint_sample.force_world[1],
                        0.0,
                    ]
                )
                if self.controller_overlays["height"]:
                    sim._add_force_arrow(
                        renderer.scene,
                        self.endpoint_sample.endpoint_position,
                        vertical_force,
                        HEIGHT_ARROW_COLOR,
                        scale=self.args.force_arrow_scale,
                        max_length=self.args.force_arrow_max_length,
                    )
                if self.controller_overlays["velocity"]:
                    sim._add_force_arrow(
                        renderer.scene,
                        self.endpoint_sample.endpoint_position,
                        horizontal_force,
                        VELOCITY_ARROW_COLOR,
                        scale=self.args.force_arrow_scale,
                        max_length=self.args.force_arrow_max_length,
                    )
                if self.controller_overlays["torque"]:
                    sim.update_yaw_visuals(
                        renderer.scene,
                        self.data,
                        self.model,
                        self.yaw_sample,
                        self.args,
                        False,
                        True,
                    )
        rgb = renderer.render()
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not ok:
            raise RuntimeError("MuJoCo frame JPEG encoding failed")
        return encoded.tobytes()

    def _publish_frame(self, frame: bytes) -> None:
        with self.frame_condition:
            self.latest_frame = frame
            self.frame_sequence += 1
            self.frame_condition.notify_all()

    def _run(self) -> None:
        renderer = None
        camera = None
        try:
            if self.enable_rendering:
                renderer = mujoco.Renderer(
                    self.model,
                    height=self.render_height,
                    width=self.render_width,
                )
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(camera)

            next_step = time.perf_counter()
            next_frame = next_step
            while not self.stop_requested:
                if self.paused:
                    time.sleep(0.02)
                    next_step = time.perf_counter()
                    next_frame = next_step
                    continue

                with self.lock:
                    self._step_unlocked()
                    now = time.perf_counter()
                    if renderer is not None and camera is not None and now >= next_frame:
                        self._publish_frame(self._render_unlocked(renderer, camera))
                        next_frame = now + self.render_period

                next_step += self.model.opt.timestep
                delay = next_step - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -0.15:
                    next_step = time.perf_counter()
        except Exception as exc:  # surfaced verbatim through /api/state
            self.runtime_error = f"{type(exc).__name__}: {exc}"
            self.paused = True
            self._publish_frame(self._placeholder_frame(self.runtime_error))
        finally:
            if renderer is not None:
                renderer.close()

    def command(self, command: str) -> dict[str, Any]:
        command = command.strip().lower()
        with self.lock:
            if command in {"w", "s", "a", "d"}:
                velocity = (
                    self.endpoint_controller.requested_velocity_local_xy.copy()
                )
                if command == "w":
                    velocity[0] += 0.05
                elif command == "s":
                    velocity[0] -= 0.05
                elif command == "a":
                    velocity[1] += 0.05
                else:
                    velocity[1] -= 0.05
                self.endpoint_controller.set_local_targets(
                    velocity_local_xy=velocity
                )
            elif command in {"i", "k"}:
                delta = 0.01 if command == "i" else -0.01
                self.endpoint_controller.set_local_targets(
                    height=self.endpoint_controller.requested_height + delta
                )
            elif command in {"j", "l"}:
                delta = sim.YAW_INCREMENT if command == "j" else -sim.YAW_INCREMENT
                self.yaw_controller.increment_target_yaw_world(delta)
            elif command == "c":
                self.yaw_controller.set_target_yaw_world(sim.DEFAULT_TARGET_YAW)
            elif command == "x":
                self.endpoint_controller.set_local_targets(
                    velocity_local_xy=(0.0, 0.0)
                )
            elif command == "v":
                enabled = not any(self.controller_overlays.values())
                for name in self.controller_overlays:
                    self.controller_overlays[name] = enabled
            elif command == "r":
                self._reset_unlocked()
            elif command in {"pause", "q"}:
                self.paused = True
            elif command in {"resume", "play"}:
                self.paused = False
            elif command == "toggle-pause":
                self.paused = not self.paused
            elif command == "toggle-controllers":
                self.controllers_enabled = not self.controllers_enabled
            else:
                raise ValueError(f"Unknown command: {command}")
            return self._state_unlocked()

    def set_velocity_target(self, *, vx: float, vy: float) -> dict[str, Any]:
        """Set the persistent robot-heading planar velocity target."""
        vx = float(vx)
        vy = float(vy)
        if not math.isfinite(vx) or not math.isfinite(vy):
            raise ValueError("velocity targets must be finite numbers")
        vx = float(np.clip(vx, *sim.VELOCITY_X_COMMAND_RANGE))
        vy = float(np.clip(vy, *sim.VELOCITY_Y_COMMAND_RANGE))
        with self.lock:
            self.endpoint_controller.set_local_targets(
                velocity_local_xy=(vx, vy)
            )
            return self._state_unlocked()

    def set_height_target(self, *, height: float) -> dict[str, Any]:
        """Set the persistent world-height target within the deployment range."""
        height = float(height)
        if not math.isfinite(height):
            raise ValueError("height target must be a finite number")
        height = float(np.clip(height, *sim.HEIGHT_COMMAND_RANGE))
        with self.lock:
            self.endpoint_controller.set_local_targets(height=height)
            return self._state_unlocked()

    def set_yaw_target(self, *, yaw_deg: float) -> dict[str, Any]:
        """Set the target bar-vector yaw in the world XY projection."""
        yaw_deg = float(yaw_deg)
        if not math.isfinite(yaw_deg):
            raise ValueError("yaw target must be a finite number")
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
        with self.lock:
            self.yaw_controller.set_target_yaw_world(math.radians(yaw_deg))
            return self._state_unlocked()

    def reset_command_targets(self) -> dict[str, Any]:
        """Restore website command defaults without resetting physical state."""
        with self.lock:
            self.endpoint_controller.set_local_targets(
                height=DEFAULT_HEIGHT,
                velocity_local_xy=(0.0, 0.0),
            )
            self.yaw_controller.set_target_yaw_world(
                math.radians(DEFAULT_YAW_DEG)
            )
            return self._state_unlocked()

    def toggle_controller_overlay(self, controller: str) -> dict[str, Any]:
        controller = controller.strip().lower()
        if controller not in self.controller_overlays:
            raise ValueError(
                "controller must be one of: velocity, height, torque"
            )
        with self.lock:
            self.controller_overlays[controller] = not self.controller_overlays[
                controller
            ]
            return self._state_unlocked()

    def move_camera(self, *, dx: float = 0.0, dy: float = 0.0, zoom: float = 0.0) -> dict[str, Any]:
        with self.lock:
            self.camera.azimuth = (self.camera.azimuth - 0.22 * dx) % 360.0
            self.camera.elevation = float(
                np.clip(self.camera.elevation - 0.18 * dy, -60.0, 18.0)
            )
            self.camera.distance = float(
                np.clip(self.camera.distance * math.exp(0.0012 * zoom), 1.6, 6.0)
            )
            return self._state_unlocked()

    def toggle_camera_follow(self) -> dict[str, Any]:
        with self.lock:
            self.camera.follow = not self.camera.follow
            return self._state_unlocked()

    def _state_unlocked(self) -> dict[str, Any]:
        endpoint_position, endpoint_velocity = self.endpoint_controller.endpoint_state(
            self.model, self.data
        )
        endpoint_velocity_local = self.endpoint_controller.world_to_local_xy(
            self.data, endpoint_velocity[:2]
        )
        velocity_reference_local = (
            self.endpoint_controller.velocity_reference_local_xy(self.data)
        )
        measured_yaw = self.yaw_controller.measured_yaw_world(self.data)
        yaw_reference = self.yaw_controller.reference_yaw_world
        yaw_requested = self.yaw_controller.requested_yaw_world
        force = (
            self.endpoint_sample.force_world.copy()
            if self.endpoint_sample is not None and self.controllers_enabled
            else np.zeros(3)
        )
        force_local_xy = self.endpoint_controller.world_to_local_xy(
            self.data, force[:2]
        )
        torque = (
            self.yaw_sample.torque_world.copy()
            if self.yaw_sample is not None and self.controllers_enabled
            else np.zeros(3)
        )
        base_position = self.evaluator.base_position
        finite = bool(
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
        )
        return {
            "connected": True,
            "paused": self.paused,
            "controllers_enabled": self.controllers_enabled,
            "vectors_visible": any(self.controller_overlays.values()),
            "overlays": self.controller_overlays.copy(),
            "runtime_error": self.runtime_error,
            "healthy": finite and self.evaluator.base_height > 0.45,
            "time": float(self.data.time),
            "base": {
                "x": float(base_position[0]),
                "y": float(base_position[1]),
                "height": float(base_position[2]),
            },
            "target": {
                "height": float(self.endpoint_controller.requested_height),
                "vx": float(
                    self.endpoint_controller.requested_velocity_local_xy[0]
                ),
                "vy": float(
                    self.endpoint_controller.requested_velocity_local_xy[1]
                ),
                "yaw_deg": math.degrees(yaw_requested),
                "vector": [math.cos(yaw_requested), math.sin(yaw_requested), 0.0],
            },
            "reference": {
                "height": float(self.endpoint_controller.height_reference),
                "vx": float(velocity_reference_local[0]),
                "vy": float(velocity_reference_local[1]),
                "yaw_deg": math.degrees(yaw_reference),
            },
            "actual": {
                "height": float(endpoint_position[2]),
                "vx": float(endpoint_velocity_local[0]),
                "vy": float(endpoint_velocity_local[1]),
                "yaw_deg": math.degrees(measured_yaw),
            },
            "wrench": {
                "fx": float(force[0]),
                "fy": float(force[1]),
                "fz": float(force[2]),
                "tx": float(torque[0]),
                "ty": float(torque[1]),
                "tz": float(torque[2]),
            },
            "wrench_local": {
                "fx": float(force_local_xy[0]),
                "fy": float(force_local_xy[1]),
                "fz": float(force[2]),
                "tz": float(torque[2]),
            },
            "limits": {
                "height": list(sim.HEIGHT_COMMAND_RANGE),
                "vx": list(sim.VELOCITY_X_COMMAND_RANGE),
                "vy": list(sim.VELOCITY_Y_COMMAND_RANGE),
            },
            "rates": {
                "physics_hz": PHYSICS_FREQUENCY,
                "controller_hz": CONTROLLER_FREQUENCY,
                "policy_hz": POLICY_FREQUENCY,
            },
            "command_frame": "robot_heading",
            "controller": {
                "height_kp": self.endpoint_controller.config.height_kp,
                "height_kd": self.endpoint_controller.config.height_kd,
                "height_force_limit": self.endpoint_controller.config.height_force_limit,
                "velocity_kp": self.endpoint_controller.config.velocity_kp,
                "velocity_ki": self.endpoint_controller.config.velocity_ki,
                "velocity_kd": self.endpoint_controller.config.velocity_kd,
                "horizontal_force_limit": self.endpoint_controller.config.horizontal_force_limit,
                "yaw_kp": self.yaw_controller.config.kp,
                "yaw_kd": self.yaw_controller.config.kd,
                "yaw_torque_limit": self.yaw_controller.config.torque_limit,
            },
            "camera_follow": self.camera.follow,
            "camera": {
                "azimuth": self.camera.azimuth,
                "elevation": self.camera.elevation,
            },
            "checkpoint": self.policy_path.name,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state_unlocked()

    def wait_for_frame(self, sequence: int, timeout: float = 2.0) -> tuple[int, bytes]:
        with self.frame_condition:
            if self.frame_sequence <= sequence:
                self.frame_condition.wait(timeout=timeout)
            return self.frame_sequence, self.latest_frame


class SimulationNotReady(RuntimeError):
    """Raised when an interactive request arrives during lazy setup."""


class SimulationManager:
    """Lazily build MuJoCo after the browser has received the demo page."""

    def __init__(self, factory: Callable[[], ColaSimulation]) -> None:
        self.factory = factory
        self.condition = threading.Condition()
        self.simulation: ColaSimulation | None = None
        self.error: str | None = None
        self.initializer: threading.Thread | None = None
        self.stop_requested = False
        self.loading_frame = ColaSimulation._placeholder_frame(
            "Loading model, controllers, and student policy..."
        )
        self.loading_sequence = 0

    def start(self) -> None:
        with self.condition:
            if self.initializer is not None or self.simulation is not None:
                return
            self.initializer = threading.Thread(
                target=self._initialize,
                name="cola-mujoco-setup",
                daemon=True,
            )
            self.initializer.start()

    def _initialize(self) -> None:
        try:
            simulation = self.factory()
            with self.condition:
                if self.stop_requested:
                    return
                self.simulation = simulation
                simulation.start()
                self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
                self.condition.notify_all()

    def require_simulation(self) -> ColaSimulation:
        self.start()
        with self.condition:
            if self.simulation is not None:
                return self.simulation
            if self.error is not None:
                raise SimulationNotReady(self.error)
            raise SimulationNotReady("MuJoCo is still loading")

    def state(self) -> dict[str, Any]:
        self.start()
        with self.condition:
            if self.simulation is not None:
                state = self.simulation.state()
                state["ready"] = True
                return state
            return {
                "ready": False,
                "connected": self.error is None,
                "runtime_error": self.error,
                "loading_message": self.error
                or "Loading the model, controllers, and student policy",
            }

    def wait_for_frame(
        self, sequence: int, timeout: float = 0.4
    ) -> tuple[int, bytes]:
        self.start()
        with self.condition:
            if self.simulation is None and self.error is None:
                self.condition.wait(timeout=timeout)
            simulation = self.simulation
            if simulation is None:
                self.loading_sequence += 1
                return self.loading_sequence, self.loading_frame
        return simulation.wait_for_frame(sequence, timeout=timeout)

    def stop(self) -> None:
        with self.condition:
            self.stop_requested = True
            simulation = self.simulation
            initializer = self.initializer
            self.condition.notify_all()
        if simulation is not None:
            simulation.stop()
        if initializer is not None and initializer is not threading.current_thread():
            initializer.join(timeout=4.0)


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "COLADemo/0.1"

    @property
    def simulation(self) -> ColaSimulation:
        return self.manager.require_simulation()

    @property
    def manager(self) -> SimulationManager:
        return self.server.simulation_manager  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        if self.path != "/api/state":
            super().log_message(format, *args)

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, value: Any, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(value, separators=(",", ":"), allow_nan=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def _send_redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_site_asset(self, request_path: str) -> None:
        static_root = (SITE_ROOT / "static").resolve()
        candidate = (SITE_ROOT / request_path.lstrip("/")).resolve()
        if static_root not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send_bytes(candidate.read_bytes(), content_type)

    def do_GET(self) -> None:  # noqa: N802
        request_path = self.path.partition("?")[0]
        if request_path in {"/", "/index.html"}:
            self._send_bytes(
                (SITE_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif request_path == "/demo":
            self._send_redirect("/demo/")
        elif request_path in {"/demo/", "/demo/index.html"}:
            self.manager.start()
            self._send_bytes(
                (DEMO_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif request_path == "/demo/demo.css":
            self._send_bytes(
                (DEMO_ROOT / "demo.css").read_bytes(), "text/css; charset=utf-8"
            )
        elif request_path == "/demo/demo.js":
            self._send_bytes(
                (DEMO_ROOT / "demo.js").read_bytes(),
                "text/javascript; charset=utf-8",
            )
        elif request_path.startswith("/static/"):
            self._send_site_asset(request_path)
        elif request_path == "/api/state":
            self._send_json(self.manager.state())
        elif request_path == "/api/frame.jpg":
            _, frame = self.manager.wait_for_frame(-1, timeout=0.2)
            self._send_bytes(frame, "image/jpeg")
        elif request_path == "/api/stream":
            self._stream_frames()
        elif request_path == "/favicon.ico":
            self._send_site_asset("/static/images/favicon.ico")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > 16_384:
            raise ValueError("Request body is too large")
        payload = self.rfile.read(length)
        value = json.loads(payload or b"{}")
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/command":
                command = payload.get("command")
                if not isinstance(command, str):
                    raise ValueError("command must be a string")
                self._send_json(self.simulation.command(command))
            elif self.path == "/api/velocity":
                if "vx" not in payload or "vy" not in payload:
                    raise ValueError("vx and vy are required")
                self._send_json(
                    self.simulation.set_velocity_target(
                        vx=float(payload["vx"]),
                        vy=float(payload["vy"]),
                    )
                )
            elif self.path == "/api/height":
                if "height" not in payload:
                    raise ValueError("height is required")
                self._send_json(
                    self.simulation.set_height_target(
                        height=float(payload["height"]),
                    )
                )
            elif self.path == "/api/yaw":
                if "yaw_deg" not in payload:
                    raise ValueError("yaw_deg is required")
                self._send_json(
                    self.simulation.set_yaw_target(
                        yaw_deg=float(payload["yaw_deg"]),
                    )
                )
            elif self.path == "/api/default-targets":
                self._send_json(self.simulation.reset_command_targets())
            elif self.path == "/api/controller-overlay":
                controller = payload.get("controller")
                if not isinstance(controller, str):
                    raise ValueError("controller must be a string")
                self._send_json(
                    self.simulation.toggle_controller_overlay(controller)
                )
            elif self.path == "/api/camera":
                self._send_json(
                    self.simulation.move_camera(
                        dx=float(payload.get("dx", 0.0)),
                        dy=float(payload.get("dy", 0.0)),
                        zoom=float(payload.get("zoom", 0.0)),
                    )
                )
            elif self.path == "/api/camera-follow":
                self._send_json(self.simulation.toggle_camera_follow())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except SimulationNotReady as exc:
            self._send_json({"error": str(exc), "ready": False}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _stream_frames(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={JPEG_BOUNDARY.decode()}",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        sequence = -1
        try:
            while not self.manager.stop_requested:
                sequence, frame = self.manager.wait_for_frame(sequence)
                self.wfile.write(b"--" + JPEG_BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], manager: SimulationManager):
        super().__init__(address, DemoHandler)
        self.simulation_manager = manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-width", type=int, default=3840)
    parser.add_argument("--render-height", type=int, default=2160)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--controllers-disabled", action="store_true")
    parser.add_argument("--no-render", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = SimulationManager(
        lambda: ColaSimulation(
            args.policy,
            device=args.device,
            render_width=args.render_width,
            render_height=args.render_height,
            render_fps=args.render_fps,
            controllers_enabled=not args.controllers_disabled,
            enable_rendering=not args.no_render,
        )
    )
    server = DemoServer((args.host, args.port), manager)

    def request_shutdown(_signum=None, _frame=None) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(f"COLA project page: http://{args.host}:{args.port}/")
    print(f"MuJoCo demo: http://{args.host}:{args.port}/demo/")
    print(f"Policy: {args.policy.resolve()}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        manager.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
