#!/usr/bin/env python3
"""Evaluate a COLA student with centered fixed bar and three-part command."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import select
import sys
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
BASE_PIPELINE = ROOT.parent / "mujoco_centered_bar_controller_sim2sim"
sys.path.insert(0, str(BASE_PIPELINE))

from controller import load_model  # noqa: E402
from run_demo import (  # noqa: E402
    _add_force_arrow,
    make_config,
    parser as endpoint_parser,
    update_bar_vector_arrow,
    update_force_arrows,
)
from run_sim2sim import (  # noqa: E402
    BAR_INITIAL_CENTER,
    CENTERED_FIXED_BAR_MODEL,
    centered_bar_sites,
    write_csv_rows,
)

from sim2sim_alignment import (  # noqa: E402
    HEIGHT_COMMAND_RANGE,
    VELOCITY_X_COMMAND_RANGE,
    VELOCITY_Y_COMMAND_RANGE,
    RobotFrameCommandEndpointForceController as EndpointForceController,
    Smokv3spStudentEvaluator as StudentEvaluator,
)

from yaw_torque_controller import (  # noqa: E402
    BarYawTorqueController,
    YawTorqueControllerConfig,
)


DEFAULT_LOOP_FIXED_MIDDLE_POLICY = (
    ROOT.parent.parent
    / "models"
    / "policy_student_loop_static_three_jitter_phase3_model_5000.jit"
)
POLICY_FREQUENCY = 50.0
CONTROLLER_FREQUENCY = 400.0
YAW_INCREMENT = math.radians(5.0)
DEFAULT_TARGET_YAW = -0.5 * math.pi

CURRENT_VECTOR_COLOR = np.array([0.05, 1.0, 0.90, 0.95], dtype=np.float32)
TARGET_VECTOR_COLOR = np.array([1.0, 0.85, 0.05, 0.95], dtype=np.float32)
YAW_TORQUE_COLOR = np.array([1.0, 0.35, 0.05, 0.95], dtype=np.float32)

TERMINAL_HELP = """Terminal commands (press Enter after each command):
  W / S            add/subtract 0.05 m/s robot-forward velocity
  A / D            add/subtract 0.05 m/s robot-left velocity
  I / K            add/subtract 0.01 m from target height
  J / L            add/subtract 5 deg from target-vector yaw
  C                reset target-vector yaw to world-frame -90 deg
  X                set both horizontal velocity targets to zero
  V [ON|OFF]       toggle or set current/target bar-vector arrows
  R                reset robot, bar, policy history, and controllers
  help             show this command list
  Q                close the simulation

Conservative sim2sim limits (enforced for every input path):
  height            0.55 to 0.85 m
  robot-local vx/vy -0.50 to +0.50 m/s per axis
"""


def parser() -> argparse.ArgumentParser:
    result = endpoint_parser()
    result.description = (
        "Centered fixed-bar sim2sim with center-point height/velocity control "
        "and world-frame bar-vector yaw control by projected world-Z torque."
    )
    result.set_defaults(
        model=CENTERED_FIXED_BAR_MODEL,
        height=0.70,
        duration=0.0,
    )
    model_action = next(action for action in result._actions if action.dest == "model")
    model_action.help = "Centered left-weld/right-point fixed-bar MuJoCo model."
    result.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_LOOP_FIXED_MIDDLE_POLICY,
        help="Exported Loop fixed-middle phase-3 student JIT.",
    )
    result.add_argument("--device", default="auto")
    result.add_argument(
        "--simulation-frequency",
        type=float,
        default=1000.0,
        help="MuJoCo and controller frequency in Hz (default: 1000).",
    )
    result.add_argument(
        "--target-yaw-deg",
        type=float,
        default=math.degrees(DEFAULT_TARGET_YAW),
        help="Requested bar-vector yaw in the world frame.",
    )
    result.add_argument("--yaw-kp", type=float, default=40.0)
    result.add_argument("--yaw-kd", type=float, default=4.0)
    result.add_argument("--yaw-torque-limit", type=float, default=10.0)
    result.add_argument("--yaw-target-rate-deg-s", type=float, default=45.0)
    result.add_argument(
        "--step-yaw-deg",
        type=float,
        default=None,
        help="Absolute world-frame yaw request applied at --step-command-time.",
    )
    result.add_argument(
        "--yaw-torque-arrow-scale",
        type=float,
        default=0.04,
        help="Displayed metres per N*m for the projected world-Z torque arrow.",
    )
    result.add_argument("--yaw-torque-arrow-max-length", type=float, default=0.6)
    result.add_argument("--target-vector-display-length", type=float, default=0.8)
    result.add_argument(
        "--enable-bar-controllers",
        action="store_true",
        help=(
            "Apply the height, horizontal-velocity, and yaw controller wrenches. "
            "They are disabled by default for the current no-external-force diagnostic."
        ),
    )
    return result


def target_status(
    endpoint_controller: EndpointForceController,
    yaw_controller: BarYawTorqueController,
) -> str:
    vector = np.array(
        [
            math.cos(yaw_controller.requested_yaw_world),
            math.sin(yaw_controller.requested_yaw_world),
            0.0,
        ]
    )
    return (
        "Requested targets: "
        f"height={endpoint_controller.requested_height:.3f} m, "
        f"robot-local velocity=("
        f"{endpoint_controller.requested_velocity_local_xy[0]:+.3f}, "
        f"{endpoint_controller.requested_velocity_local_xy[1]:+.3f}) m/s, "
        f"world_vector_yaw={math.degrees(yaw_controller.requested_yaw_world):+.1f} deg, "
        f"target-vector=({vector[0]:+.3f}, {vector[1]:+.3f}, {vector[2]:+.3f})"
        f"; limits: height={HEIGHT_COMMAND_RANGE}, "
        f"vx={VELOCITY_X_COMMAND_RANGE}, vy={VELOCITY_Y_COMMAND_RANGE}"
    )


class TerminalCommandInput:
    def __init__(
        self,
        endpoint_controller: EndpointForceController,
        yaw_controller: BarYawTorqueController,
        reset_callback,
        help_text: str = TERMINAL_HELP,
        perturb_callback=None,
    ) -> None:
        self.endpoint_controller = endpoint_controller
        self.yaw_controller = yaw_controller
        self.reset_callback = reset_callback
        self.help_text = help_text
        self.perturb_callback = perturb_callback
        self.show_vectors = True
        self.enabled = True
        self.quit_requested = False

    @staticmethod
    def prompt() -> None:
        print("cola> ", end="", flush=True)

    def start(self) -> None:
        print(self.help_text, end="")
        print(target_status(self.endpoint_controller, self.yaw_controller))
        self.prompt()

    def _execute(self, line: str) -> None:
        tokens = line.strip().lower().split()
        if not tokens:
            return
        command = tokens[0]
        if command in {"w", "s", "a", "d", "i", "k", "j", "l", "c", "x", "p", "r", "q", "help"} and len(tokens) != 1:
            raise ValueError(f"{command.upper()} takes no arguments")

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
            delta = YAW_INCREMENT if command == "j" else -YAW_INCREMENT
            self.yaw_controller.increment_target_yaw_world(delta)
        elif command == "c":
            self.yaw_controller.set_target_yaw_world(DEFAULT_TARGET_YAW)
        elif command == "x":
            self.endpoint_controller.set_local_targets(
                velocity_local_xy=(0.0, 0.0)
            )
        elif command == "p":
            if self.perturb_callback is None:
                raise ValueError("perturbation command is unavailable")
            self.perturb_callback()
            return
        elif command == "v":
            if len(tokens) > 2:
                raise ValueError("usage: V [ON|OFF]")
            if len(tokens) == 1:
                self.show_vectors = not self.show_vectors
            elif tokens[1] in {"on", "off"}:
                self.show_vectors = tokens[1] == "on"
            else:
                raise ValueError("usage: V [ON|OFF]")
            print(f"Bar-vector display: {'ON' if self.show_vectors else 'OFF'}")
            return
        elif command == "r":
            self.reset_callback()
        elif command == "help":
            print(self.help_text, end="")
            return
        elif command == "q":
            self.quit_requested = True
            return
        else:
            raise ValueError(f"unknown command {tokens[0]!r}; type 'help'")
        print(target_status(self.endpoint_controller, self.yaw_controller))

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
            print("\nTerminal input closed; viewer remains active.")
            return
        try:
            self._execute(line)
        except ValueError as error:
            print(f"Command error: {error}")
        if not self.quit_requested:
            self.prompt()


def _row(endpoint_sample, yaw_sample, evaluator: StudentEvaluator) -> dict[str, float]:
    return {
        "time": endpoint_sample.time,
        "height_ref": endpoint_sample.height_reference,
        "height": endpoint_sample.endpoint_position[2],
        "vx_ref": endpoint_sample.velocity_reference_xy[0],
        "vx": endpoint_sample.endpoint_velocity[0],
        "vy_ref": endpoint_sample.velocity_reference_xy[1],
        "vy": endpoint_sample.endpoint_velocity[1],
        "fx": endpoint_sample.force_world[0],
        "fy": endpoint_sample.force_world[1],
        "fz": endpoint_sample.force_world[2],
        "yaw_requested_world_deg": math.degrees(yaw_sample.requested_yaw_world),
        "yaw_reference_world_deg": math.degrees(yaw_sample.reference_yaw_world),
        "yaw_measured_world_deg": math.degrees(yaw_sample.measured_yaw_world),
        "yaw_error_deg": math.degrees(yaw_sample.yaw_error),
        "yaw_reference_rate_deg_s": math.degrees(yaw_sample.reference_rate),
        "yaw_rate_world_deg_s": math.degrees(yaw_sample.yaw_rate_world),
        "yaw_p_torque": yaw_sample.p_torque,
        "yaw_d_torque": yaw_sample.d_torque,
        "yaw_control_torque_scalar": yaw_sample.control_torque_scalar,
        "yaw_axis_world_x": yaw_sample.yaw_axis_world[0],
        "yaw_axis_world_y": yaw_sample.yaw_axis_world[1],
        "yaw_axis_world_z": yaw_sample.yaw_axis_world[2],
        "yaw_torque_x": yaw_sample.torque_world[0],
        "yaw_torque_y": yaw_sample.torque_world[1],
        "yaw_torque_z": yaw_sample.torque_world[2],
        "yaw_torque_axial": yaw_sample.axial_torque,
        "yaw_torque_saturated": int(yaw_sample.saturated),
        "bar_vector_world_x": yaw_sample.current_vector_world[0],
        "bar_vector_world_y": yaw_sample.current_vector_world[1],
        "bar_vector_world_z": yaw_sample.current_vector_world[2],
        "target_vector_world_x": yaw_sample.target_vector_world[0],
        "target_vector_world_y": yaw_sample.target_vector_world[1],
        "target_vector_world_z": yaw_sample.target_vector_world[2],
        "base_x": evaluator.base_position[0],
        "base_y": evaluator.base_position[1],
        "base_z": evaluator.base_position[2],
    }


def _apply_scripted_step(args, endpoint_controller, yaw_controller, applied, sim_time):
    if applied or args.step_command_time is None or sim_time < args.step_command_time:
        return applied
    height = endpoint_controller.requested_height if args.step_height is None else args.step_height
    velocity = endpoint_controller.requested_velocity_local_xy.copy()
    if args.step_vx is not None:
        velocity[0] = args.step_vx
    if args.step_vy is not None:
        velocity[1] = args.step_vy
    endpoint_controller.set_local_targets(
        height=height, velocity_local_xy=velocity
    )
    if args.step_yaw_deg is not None:
        yaw_controller.set_target_yaw_world(math.radians(args.step_yaw_deg))
    print(f"Applied scripted targets at t={sim_time:.3f}: {target_status(endpoint_controller, yaw_controller)}")
    return True


def update_yaw_visuals(
    scene: mujoco.MjvScene,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    yaw_sample,
    args,
    show_vectors: bool,
    show_torque: bool,
) -> None:
    center = data.site_xpos[model.site("controller_point").id].copy()
    if show_torque and getattr(args, "show_yaw_torque_arrow", True):
        _add_force_arrow(
            scene,
            center,
            yaw_sample.torque_world,
            YAW_TORQUE_COLOR,
            scale=args.yaw_torque_arrow_scale,
            max_length=args.yaw_torque_arrow_max_length,
        )
    if not show_vectors:
        return
    current_vector = yaw_sample.current_vector_world
    current_norm = float(np.linalg.norm(current_vector))
    if current_norm > 1.0e-9 and scene.ngeom < scene.maxgeom:
        current_end = (
            center
            + args.target_vector_display_length * current_vector / current_norm
        )
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            CURRENT_VECTOR_COLOR,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            args.bar_vector_radius,
            center,
            current_end,
        )
        scene.ngeom += 1
    target_end = center + args.target_vector_display_length * yaw_sample.target_vector_world
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        TARGET_VECTOR_COLOR,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        args.bar_vector_radius,
        center,
        target_end,
    )
    scene.ngeom += 1


def summarize(rows: list[dict[str, float]], step_time: float | None) -> dict[str, float]:
    if not rows:
        return {}
    selected = rows
    if step_time is not None:
        selected = [row for row in rows if row["time"] >= step_time]
    if not selected:
        selected = rows
    tail_size = max(1, int(0.2 * len(selected)))
    tail = selected[-tail_size:]
    result = {
        "final_mean_abs_yaw_error_deg": float(np.mean([abs(row["yaw_error_deg"]) for row in tail])),
        "max_abs_yaw_error_deg": float(max(abs(row["yaw_error_deg"]) for row in selected)),
        "max_abs_yaw_torque_nm": float(max(abs(row["yaw_torque_z"]) for row in selected)),
        "torque_saturation_fraction": float(np.mean([row["yaw_torque_saturated"] for row in selected])),
        "minimum_base_height_m": float(min(row["base_z"] for row in selected)),
        "maximum_base_xy_drift_m": float(max(math.hypot(row["base_x"], row["base_y"]) for row in selected)),
        "final_bar_height_m": float(tail[-1]["height"]),
    }
    return result


def main() -> int:
    args = parser().parse_args()
    if args.headless and args.duration <= 0.0:
        raise SystemExit("--headless requires a positive --duration")
    if args.simulation_frequency <= 0.0:
        raise SystemExit("--simulation-frequency must be positive")
    if args.simulation_frequency < CONTROLLER_FREQUENCY:
        raise SystemExit(
            f"--simulation-frequency must be at least {CONTROLLER_FREQUENCY:g} Hz"
        )
    if not args.model.is_file():
        raise SystemExit(f"MuJoCo model not found: {args.model}")
    if not args.policy.is_file():
        raise SystemExit(f"Student policy not found: {args.policy}")

    model, data = load_model(str(args.model.resolve()))
    model.opt.timestep = 1.0 / args.simulation_frequency
    controller_dt = 1.0 / CONTROLLER_FREQUENCY
    policy_decimation_float = args.simulation_frequency / POLICY_FREQUENCY
    policy_decimation = int(round(policy_decimation_float))
    if not math.isclose(policy_decimation_float, policy_decimation, abs_tol=1e-12):
        raise SystemExit("simulation frequency must be an integer multiple of 50 Hz")

    evaluator = StudentEvaluator(model, data, args.policy, args.device)
    endpoint_controller = EndpointForceController(
        model,
        make_config(args, dt=controller_dt),
        site_name="controller_point",
    )
    endpoint_controller.set_local_targets(
        height=args.height, velocity_local_xy=(args.vx, args.vy)
    )
    endpoint_controller.reset(model, data)
    yaw_controller = BarYawTorqueController(
        model,
        YawTorqueControllerConfig(
            dt=controller_dt,
            target_yaw_world=math.radians(args.target_yaw_deg),
            kp=args.yaw_kp,
            kd=args.yaw_kd,
            torque_limit=args.yaw_torque_limit,
            target_rate_limit=math.radians(args.yaw_target_rate_deg_s),
        ),
    )
    yaw_controller.set_target_yaw_world(math.radians(args.target_yaw_deg))
    yaw_controller.reset(model, data)

    physics_step = 0
    controller_phase = 0.0
    step_applied = False
    last_report_time = -1.0
    rows: list[dict[str, float]] = []
    endpoint_sample = None
    yaw_sample = None

    def reset_all() -> None:
        nonlocal controller_phase, endpoint_sample, physics_step, step_applied, yaw_sample, last_report_time
        evaluator.reset_simulation()
        endpoint_controller.reset(model, data)
        yaw_controller.reset(model, data)
        physics_step = 0
        controller_phase = 0.0
        endpoint_sample = None
        yaw_sample = None
        step_applied = False
        last_report_time = -1.0
        print("Reset robot, centered fixed bar, policy history, and all controllers.")

    print(f"Policy: {args.policy.resolve()}")
    print(f"MuJoCo physics: {args.simulation_frequency:g} Hz")
    print(
        f"All three bar controllers: {CONTROLLER_FREQUENCY:g} Hz "
        "(sample-and-hold between physics steps)"
    )
    print(f"Student policy: {POLICY_FREQUENCY:g} Hz (decimation {policy_decimation})")
    print(f"Centered bar reset: {np.round(BAR_INITIAL_CENTER, 3)} m")
    print("Attachments: left weld plus right point constraint; no end plates")
    print("Height/velocity force point and yaw-torque reference point: bar center")
    print(
        "Bar external controllers: "
        + ("ENABLED" if args.enable_bar_controllers else "DISABLED (diagnostic)")
    )
    print(
        "Yaw PD: "
        f"Kp={args.yaw_kp:g} N*m/rad, Kd={args.yaw_kd:g} N*m*s/rad, "
        f"limit={args.yaw_torque_limit:g} N*m, target slew={args.yaw_target_rate_deg_s:g} deg/s"
    )
    print("Arrows: blue=vertical force, magenta=horizontal force, orange=projected yaw torque, cyan=current vector, yellow=target vector")
    print(target_status(endpoint_controller, yaw_controller))

    def run_one_step(periodic_report: bool):
        nonlocal controller_phase, endpoint_sample, physics_step, step_applied, yaw_sample, last_report_time
        step_applied = _apply_scripted_step(
            args,
            endpoint_controller,
            yaw_controller,
            step_applied,
            float(data.time),
        )
        if physics_step % policy_decimation == 0:
            evaluator.infer_action()
        data.qfrc_applied.fill(0.0)
        evaluator.apply_robot_pd()
        controller_due = physics_step == 0
        if not controller_due:
            controller_phase += CONTROLLER_FREQUENCY
            if controller_phase >= args.simulation_frequency:
                controller_phase -= args.simulation_frequency
                controller_due = True
        if controller_due:
            endpoint_sample = endpoint_controller.compute(model, data)
            yaw_sample = yaw_controller.compute(model, data)
            if not args.enable_bar_controllers:
                # Keep command/vector diagnostics available, but report and
                # render the actually applied wrench: exactly zero here.
                endpoint_sample.force_world.fill(0.0)
                yaw_sample.torque_world.fill(0.0)
                yaw_sample.control_torque_scalar = 0.0
        assert endpoint_sample is not None and yaw_sample is not None
        if args.enable_bar_controllers:
            endpoint_controller.apply(model, data, endpoint_sample)
            yaw_controller.apply(model, data, yaw_sample)
        mujoco.mj_step(model, data)
        physics_step += 1
        result = _row(endpoint_sample, yaw_sample, evaluator)
        rows.append(result)
        if periodic_report and endpoint_sample.time - last_report_time >= 0.5:
            last_report_time = endpoint_sample.time
            print(
                f"t={endpoint_sample.time:6.2f} base_z={evaluator.base_height:+.3f} "
                f"bar_h={endpoint_sample.endpoint_position[2]:+.3f} "
                f"world_yaw={math.degrees(yaw_sample.measured_yaw_world):+.1f}/"
                f"{math.degrees(yaw_sample.reference_yaw_world):+.1f} deg "
                f"Tz={yaw_sample.torque_world[2]:+.2f} N*m "
                f"F=({endpoint_sample.force_world[0]:+.1f},"
                f"{endpoint_sample.force_world[1]:+.1f},"
                f"{endpoint_sample.force_world[2]:+.1f}) N"
            )
        return endpoint_sample, yaw_sample

    if args.headless:
        while data.time < args.duration:
            run_one_step(periodic_report=True)
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            terminal = TerminalCommandInput(
                endpoint_controller,
                yaw_controller,
                reset_all,
            )
            terminal.start()
            wall_start = time.perf_counter()
            while (
                viewer.is_running()
                and not terminal.quit_requested
                and (args.duration <= 0.0 or data.time < args.duration)
            ):
                step_start = time.perf_counter()
                terminal.poll()
                endpoint_sample, yaw_sample = run_one_step(periodic_report=False)
                with viewer.lock():
                    if args.enable_bar_controllers:
                        update_force_arrows(
                            viewer.user_scn,
                            endpoint_sample,
                            scale=args.force_arrow_scale,
                            max_length=args.force_arrow_max_length,
                        )
                    update_yaw_visuals(
                        viewer.user_scn,
                        data,
                        model,
                        yaw_sample,
                        args,
                        terminal.show_vectors,
                        args.enable_bar_controllers,
                    )
                viewer.sync()
                if args.realtime_factor > 0.0:
                    target_wall_dt = model.opt.timestep / args.realtime_factor
                    remaining = target_wall_dt - (time.perf_counter() - step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
            print(f"Wall time: {time.perf_counter() - wall_start:.2f} s")
            if args.csv is not None:
                write_csv_rows(args.csv, rows)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    if args.csv is not None:
        write_csv_rows(args.csv, rows)
    metrics = summarize(rows, args.step_command_time)
    if metrics:
        print("Summary: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
