#!/usr/bin/env python3
"""Run the standalone COLA partner-support endpoint controller in MuJoCo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import select
import sys
import time

import mujoco
import numpy as np

from controller import EndpointControllerConfig, EndpointForceController, load_model


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "carried_object.xml"

TERMINAL_HELP = """Terminal commands (press Enter after each command):
  W                add 0.05 m/s to the X velocity target
  S                subtract 0.05 m/s from the X velocity target
  A                add 0.05 m/s to the Y velocity target
  D                subtract 0.05 m/s from the Y velocity target
  I                add 0.01 m to the height target
  K                subtract 0.01 m from the height target
  X                set both horizontal velocities to zero
  V [ON|OFF]       toggle, enable, or disable the bar endpoint vector
  R                reset the simulation and controller
  help             show this command list
  Q                close the simulation
"""

HEIGHT_ARROW_COLOR = np.array([0.10, 0.45, 1.0, 0.95], dtype=np.float32)
VELOCITY_ARROW_COLOR = np.array([1.0, 0.15, 0.80, 0.95], dtype=np.float32)
BAR_VECTOR_COLOR = np.array([0.05, 1.0, 0.90, 0.95], dtype=np.float32)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Apply world-frame height PD and horizontal velocity PID forces at "
            "the human endpoint of one free carried object."
        )
    )
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--height", type=float, default=0.65)
    result.add_argument("--vx", type=float, default=0.0)
    result.add_argument("--vy", type=float, default=0.0)
    result.add_argument("--height-kp", type=float, default=800.0)
    result.add_argument("--height-kd", type=float, default=35.0)
    result.add_argument("--velocity-kp", type=float, default=30.0)
    result.add_argument("--velocity-ki", type=float, default=60.0)
    result.add_argument("--velocity-kd", type=float, default=0.1)
    result.add_argument("--height-force-limit", type=float, default=300.0)
    result.add_argument(
        "--horizontal-force-limit",
        type=float,
        default=100.0,
        help="Maximum norm of the combined world-X/Y controller force in N.",
    )
    result.add_argument("--integral-force-limit", type=float, default=15.0)
    result.add_argument("--derivative-cutoff-hz", type=float, default=20.0)
    result.add_argument("--height-target-rate-limit", type=float, default=0.20)
    result.add_argument("--velocity-target-slew-limit", type=float, default=2.0)
    result.add_argument("--velocity-error-deadband", type=float, default=0.005)
    result.add_argument(
        "--gravity-feedforward",
        type=float,
        default=None,
        help=(
            "Fixed vertical feedforward in N; default uses the carried body's "
            "mass times world-Z gravity, matching Isaac training."
        ),
    )
    result.add_argument("--duration", type=float, default=10.0)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--realtime-factor", type=float, default=1.0)
    result.add_argument(
        "--force-arrow-scale",
        type=float,
        default=0.01,
        help="Viewer arrow length in metres per newton before visual clipping.",
    )
    result.add_argument(
        "--force-arrow-max-length",
        type=float,
        default=0.8,
        help="Maximum displayed arrow length in metres.",
    )
    result.add_argument(
        "--bar-vector-radius",
        type=float,
        default=0.018,
        help=(
            "Radius in metres of the cyan bar endpoint vector. "
            "Its displayed length is always the exact endpoint displacement."
        ),
    )
    result.add_argument("--csv", type=Path, default=None)
    result.add_argument(
        "--step-command-time",
        type=float,
        default=None,
        help="At this simulation time, apply the optional --step-* targets.",
    )
    result.add_argument("--step-height", type=float, default=None)
    result.add_argument("--step-vx", type=float, default=None)
    result.add_argument("--step-vy", type=float, default=None)
    return result


def make_config(
    args: argparse.Namespace, *, dt: float | None = None
) -> EndpointControllerConfig:
    config = EndpointControllerConfig(
        target_height=args.height,
        target_velocity_xy=np.array([args.vx, args.vy], dtype=float),
        height_kp=args.height_kp,
        height_kd=args.height_kd,
        velocity_kp=args.velocity_kp,
        velocity_ki=args.velocity_ki,
        velocity_kd=args.velocity_kd,
        height_force_limit=args.height_force_limit,
        horizontal_force_limit=args.horizontal_force_limit,
        integral_force_limit=args.integral_force_limit,
        derivative_cutoff_hz=args.derivative_cutoff_hz,
        height_target_rate_limit=args.height_target_rate_limit,
        velocity_target_slew_limit=args.velocity_target_slew_limit,
        velocity_error_deadband=args.velocity_error_deadband,
        gravity_feedforward=args.gravity_feedforward,
    )
    if dt is not None:
        config.dt = float(dt)
    return config


def reset_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: EndpointForceController,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    controller.reset(model, data)


def target_status(controller: EndpointForceController) -> str:
    return (
        f"Requested targets: height={controller.requested_height:.3f}, "
        f"vx={controller.requested_velocity_xy[0]:.3f}, "
        f"vy={controller.requested_velocity_xy[1]:.3f}"
    )


def _add_force_arrow(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    force: np.ndarray,
    color: np.ndarray,
    *,
    scale: float,
    max_length: float,
) -> None:
    magnitude = float(np.linalg.norm(force))
    if magnitude < 1e-9 or scene.ngeom >= scene.maxgeom:
        return
    display_length = min(scale * magnitude, max_length)
    end = start + force * (display_length / magnitude)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        color,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.012,
        start,
        end,
    )
    scene.ngeom += 1


def bar_endpoint_vector(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return robot endpoint, human endpoint, and robot-to-human world vector."""

    robot_endpoint = data.site_xpos[model.site("robot_endpoint").id].copy()
    human_endpoint = data.site_xpos[model.site("human_endpoint").id].copy()
    return robot_endpoint, human_endpoint, human_endpoint - robot_endpoint


def update_bar_vector_arrow(
    scene: mujoco.MjvScene,
    robot_endpoint: np.ndarray,
    human_endpoint: np.ndarray,
    *,
    radius: float,
) -> None:
    """Add the exact robot-end to human-end displacement vector to a scene."""

    if np.linalg.norm(human_endpoint - robot_endpoint) < 1e-9:
        return
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        BAR_VECTOR_COLOR,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        robot_endpoint,
        human_endpoint,
    )
    scene.ngeom += 1


def update_force_arrows(
    scene: mujoco.MjvScene,
    sample,
    *,
    scale: float,
    max_length: float,
) -> None:
    """Draw separate height and horizontal-velocity force arrows."""

    scene.ngeom = 0
    vertical_force = np.array([0.0, 0.0, sample.force_world[2]])
    horizontal_force = np.array(
        [sample.force_world[0], sample.force_world[1], 0.0]
    )
    _add_force_arrow(
        scene,
        sample.endpoint_position,
        vertical_force,
        HEIGHT_ARROW_COLOR,
        scale=scale,
        max_length=max_length,
    )
    _add_force_arrow(
        scene,
        sample.endpoint_position,
        horizontal_force,
        VELOCITY_ARROW_COLOR,
        scale=scale,
        max_length=max_length,
    )


class ViewerDisplayOptions:
    """Mutable runtime display options controlled by terminal commands."""

    def __init__(self) -> None:
        self.show_bar_vector = False

    def set_bar_vector(self, enabled: bool | None = None) -> bool:
        if enabled is None:
            enabled = not self.show_bar_vector
        self.show_bar_vector = enabled
        return self.show_bar_vector


def execute_terminal_command(
    line: str,
    controller: EndpointForceController,
    reset_callback,
    set_bar_vector_visibility=None,
) -> bool:
    """Execute one terminal command and return True when quit is requested."""

    tokens = line.strip().split()
    if not tokens:
        return False
    command = tokens[0].lower()

    def require_arguments(count: int, usage: str) -> None:
        if len(tokens) != count + 1:
            raise ValueError(f"usage: {usage}")

    try:
        if command in {"w", "s", "a", "d", "i", "k"}:
            require_arguments(0, command.upper())
            if command == "i":
                controller.set_targets(height=controller.requested_height + 0.01)
            elif command == "k":
                controller.set_targets(height=controller.requested_height - 0.01)
            else:
                velocity = controller.requested_velocity_xy.copy()
                if command == "w":
                    velocity[0] += 0.05
                elif command == "s":
                    velocity[0] -= 0.05
                elif command == "a":
                    velocity[1] += 0.05
                else:
                    velocity[1] -= 0.05
                controller.set_targets(velocity_xy=velocity)
        elif command == "x":
            require_arguments(0, "X")
            controller.set_targets(velocity_xy=(0.0, 0.0))
        elif command == "v":
            if len(tokens) > 2:
                raise ValueError("usage: V [ON|OFF]")
            if set_bar_vector_visibility is None:
                raise ValueError("bar-vector display control is unavailable")
            enabled = None
            if len(tokens) == 2:
                option = tokens[1].lower()
                if option not in {"on", "off"}:
                    raise ValueError("usage: V [ON|OFF]")
                enabled = option == "on"
            visible = set_bar_vector_visibility(enabled)
            print(f"Robot-to-human vector display: {'ON' if visible else 'OFF'}")
            return False
        elif command == "r":
            require_arguments(0, "R")
            reset_callback()
        elif command == "help":
            require_arguments(0, "help")
            print(TERMINAL_HELP, end="")
            return False
        elif command == "q":
            require_arguments(0, "Q")
            return True
        else:
            raise ValueError(f"unknown command {tokens[0]!r}; type 'help'")
    except ValueError as error:
        print(f"Command error: {error}")
        return False

    print(target_status(controller))
    return False


class TerminalCommandInput:
    """Non-blocking, line-oriented command input for the interactive viewer."""

    def __init__(
        self,
        controller: EndpointForceController,
        reset_callback,
        set_bar_vector_visibility,
    ) -> None:
        self.controller = controller
        self.reset_callback = reset_callback
        self.set_bar_vector_visibility = set_bar_vector_visibility
        self.enabled = True
        self.quit_requested = False

    @staticmethod
    def prompt() -> None:
        print("cola> ", end="", flush=True)

    def start(self) -> None:
        print(TERMINAL_HELP, end="")
        print(target_status(self.controller))
        self.prompt()

    def poll(self) -> None:
        if not self.enabled or self.quit_requested:
            return
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, ValueError) as error:
            self.enabled = False
            print(f"\nTerminal command input unavailable: {error}")
            return
        if not readable:
            return

        line = sys.stdin.readline()
        if line == "":
            self.enabled = False
            print("\nTerminal input closed; the viewer remains active.")
            return
        self.quit_requested = execute_terminal_command(
            line,
            self.controller,
            self.reset_callback,
            self.set_bar_vector_visibility,
        )
        if not self.quit_requested:
            self.prompt()


def row(
    sample,
    robot_endpoint: np.ndarray,
    human_endpoint: np.ndarray,
    robot_to_human_world: np.ndarray,
) -> dict[str, float]:
    return {
        "time": sample.time,
        "height_ref": sample.height_reference,
        "height": sample.endpoint_position[2],
        "vx_ref": sample.velocity_reference_xy[0],
        "vx": sample.endpoint_velocity[0],
        "vy_ref": sample.velocity_reference_xy[1],
        "vy": sample.endpoint_velocity[1],
        "fx": sample.force_world[0],
        "fy": sample.force_world[1],
        "fz": sample.force_world[2],
        "height_p": sample.height_p_force,
        "height_d": sample.height_d_force,
        "height_gravity": sample.height_gravity_force,
        "velocity_px": sample.velocity_p_force_xy[0],
        "velocity_py": sample.velocity_p_force_xy[1],
        "velocity_ix": sample.velocity_i_force_xy[0],
        "velocity_iy": sample.velocity_i_force_xy[1],
        "velocity_dx": sample.velocity_d_force_xy[0],
        "velocity_dy": sample.velocity_d_force_xy[1],
        "robot_endpoint_x": robot_endpoint[0],
        "robot_endpoint_y": robot_endpoint[1],
        "robot_endpoint_z": robot_endpoint[2],
        "human_endpoint_x": human_endpoint[0],
        "human_endpoint_y": human_endpoint[1],
        "human_endpoint_z": human_endpoint[2],
        "robot_to_human_world_x": robot_to_human_world[0],
        "robot_to_human_world_y": robot_to_human_world[1],
        "robot_to_human_world_z": robot_to_human_world[2],
        "robot_to_human_norm": np.linalg.norm(robot_to_human_world),
    }


def maybe_apply_step(args, controller, applied: bool, sim_time: float) -> bool:
    if applied or args.step_command_time is None or sim_time < args.step_command_time:
        return applied
    height = controller.requested_height if args.step_height is None else args.step_height
    velocity = controller.requested_velocity_xy.copy()
    if args.step_vx is not None:
        velocity[0] = args.step_vx
    if args.step_vy is not None:
        velocity[1] = args.step_vy
    controller.set_targets(height=height, velocity_xy=velocity)
    print(
        f"Applied step targets at t={sim_time:.3f}: "
        f"height={height:.3f}, velocity=({velocity[0]:.3f}, {velocity[1]:.3f})"
    )
    return True


def simulate_step(model, data, controller):
    data.qfrc_applied.fill(0.0)
    sample = controller.compute(model, data)
    controller.apply(model, data, sample)
    mujoco.mj_step(model, data)
    return sample


def main() -> int:
    args = parser().parse_args()
    model, data = load_model(str(args.model.resolve()))
    controller = EndpointForceController(model, make_config(args))
    controller.set_targets(height=args.height, velocity_xy=(args.vx, args.vy))
    reset_simulation(model, data, controller)

    print(f"MuJoCo timestep: {model.opt.timestep:.4f} s (400 Hz)")
    print("Model: one unconstrained rigid carried object")
    print("Force application site: human_endpoint on body carried_object")
    print(
        "Initial vertical gravity feedforward: "
        f"{controller.gravity_feedforward:.3f} N"
    )
    print("Force arrows: blue=height controller, magenta=velocity controller")
    print("Bar vector: enter 'V ON' to show the exact cyan robot-to-human vector")

    rows: list[dict[str, float]] = []
    step_applied = False
    last_report_time = -1.0

    def record_and_report(sample, *, periodic_report: bool = True) -> None:
        nonlocal last_report_time
        robot_endpoint, human_endpoint, robot_to_human_world = bar_endpoint_vector(
            model, data
        )
        if args.csv is not None:
            rows.append(
                row(
                    sample,
                    robot_endpoint,
                    human_endpoint,
                    robot_to_human_world,
                )
            )
        if periodic_report and sample.time - last_report_time >= 0.5:
            last_report_time = sample.time
            print(
                f"t={sample.time:6.2f}  h={sample.endpoint_position[2]:+.3f}/"
                f"{sample.height_reference:+.3f}  "
                f"vxy=({sample.endpoint_velocity[0]:+.3f},"
                f"{sample.endpoint_velocity[1]:+.3f})/"
                f"({sample.velocity_reference_xy[0]:+.3f},"
                f"{sample.velocity_reference_xy[1]:+.3f})  "
                f"F=({sample.force_world[0]:+.1f},"
                f"{sample.force_world[1]:+.1f},"
                f"{sample.force_world[2]:+.1f})  "
                f"r->h=({robot_to_human_world[0]:+.3f},"
                f"{robot_to_human_world[1]:+.3f},"
                f"{robot_to_human_world[2]:+.3f})"
            )

    if args.headless:
        while data.time < args.duration:
            step_applied = maybe_apply_step(args, controller, step_applied, data.time)
            sample = simulate_step(model, data, controller)
            record_and_report(sample)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            viewer_options = ViewerDisplayOptions()
            terminal = TerminalCommandInput(
                controller,
                lambda: reset_simulation(model, data, controller),
                viewer_options.set_bar_vector,
            )
            terminal.start()
            wall_start = time.perf_counter()
            while (
                viewer.is_running()
                and not terminal.quit_requested
                and data.time < args.duration
            ):
                step_start = time.perf_counter()
                terminal.poll()
                step_applied = maybe_apply_step(args, controller, step_applied, data.time)
                sample = simulate_step(model, data, controller)
                record_and_report(sample, periodic_report=False)
                with viewer.lock():
                    update_force_arrows(
                        viewer.user_scn,
                        sample,
                        scale=args.force_arrow_scale,
                        max_length=args.force_arrow_max_length,
                    )
                    if viewer_options.show_bar_vector:
                        robot_endpoint, human_endpoint, _ = bar_endpoint_vector(
                            model, data
                        )
                        update_bar_vector_arrow(
                            viewer.user_scn,
                            robot_endpoint,
                            human_endpoint,
                            radius=args.bar_vector_radius,
                        )
                viewer.sync()
                if args.realtime_factor > 0.0:
                    target_wall_dt = model.opt.timestep / args.realtime_factor
                    remaining = target_wall_dt - (time.perf_counter() - step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
            print(f"Wall time: {time.perf_counter() - wall_start:.2f} s")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {len(rows)} samples to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
