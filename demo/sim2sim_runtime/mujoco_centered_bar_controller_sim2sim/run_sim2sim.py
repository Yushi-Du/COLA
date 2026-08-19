#!/usr/bin/env python3
"""Evaluate a COLA student with a centered free bar and center-point control."""

from __future__ import annotations

import csv
from collections import deque
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

from controller import EndpointForceController, load_model
from run_demo import (
    TerminalCommandInput,
    ViewerDisplayOptions,
    make_config,
    maybe_apply_step,
    parser as controller_parser,
    update_bar_vector_arrow,
    update_force_arrows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "centered_bar_scene.xml"
CENTERED_FIXED_BAR_MODEL = ROOT / "model" / "centered_fixed_bar_scene.xml"
DEFAULT_POLICY = ROOT / "model" / "student_model_3999.jit"

SIM_DT = 0.001
POLICY_DECIMATION = 20
POLICY_DT = SIM_DT * POLICY_DECIMATION
N_ACTIONS = 29
OBS_FRAME_SIZE = 111
OBS_HISTORY_LENGTH = 25
ACTION_SCALE = 0.25
OBS_CLIP = 100.0
ACTION_CLIP = 100.0

ROBOT_INITIAL_POSITION = np.array([0.0, 0.0, 0.80], dtype=np.float64)
BAR_INITIAL_CENTER = np.array([0.298, 0.0, 0.924], dtype=np.float64)
BAR_NEGATIVE_Y_ENDPOINT_OFFSET = np.array([0.0, -0.8, 0.0], dtype=np.float64)
BAR_POSITIVE_Y_ENDPOINT_OFFSET = np.array([0.0, 0.8, 0.0], dtype=np.float64)

POLICY_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_POSITION_BY_NAME = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_wrist_roll_joint": -np.pi / 2.0,
    "right_wrist_roll_joint": np.pi / 2.0,
}

LEFT_PALM_QUATERNION = np.array([0.707, -0.707, 0.0, 0.0], dtype=np.float32)
RIGHT_PALM_QUATERNION = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32)
LEFT_PALM_POSITION = np.array([0.2413, 0.1517, 0.0952], dtype=np.float32)
RIGHT_PALM_POSITION = np.array([0.2413, -0.1516, 0.0952], dtype=np.float32)
MASKED_STUDENT_HEIGHT_COMMAND = 0.78


def gains_for_joint(name: str) -> tuple[float, float]:
    if "hip" in name:
        return 80.0, 2.0
    if "knee" in name:
        return 160.0, 4.0
    if "waist" in name:
        return 200.0, 5.0
    if "ankle" in name:
        return 20.0, 0.5
    if "shoulder" in name or "elbow" in name:
        return 40.0, 1.0
    if "wrist" in name:
        return 40.0, 0.5
    raise KeyError(f"No actuator gains are defined for {name!r}")


def rotate_inverse(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into a quaternion's local frame."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    vector = np.asarray(vector, dtype=np.float64)
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        - 2.0 * scalar * np.cross(imaginary, vector)
        + 2.0 * imaginary * np.dot(imaginary, vector)
    )


class StudentEvaluator:
    """Student observation history, inference, and 29-DOF position control."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: Path,
        device: str,
    ) -> None:
        self.model = model
        self.data = data
        self.device = self._resolve_device(device)

        robot_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.mujoco_joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in robot_joint_ids
        ]
        if len(self.mujoco_joint_names) != N_ACTIONS:
            raise ValueError(
                f"Expected {N_ACTIONS} robot hinge joints, found "
                f"{len(self.mujoco_joint_names)}"
            )
        if set(self.mujoco_joint_names) != set(POLICY_JOINT_ORDER):
            missing = sorted(set(POLICY_JOINT_ORDER) - set(self.mujoco_joint_names))
            extra = sorted(set(self.mujoco_joint_names) - set(POLICY_JOINT_ORDER))
            raise ValueError(f"Joint-name mismatch; missing={missing}, extra={extra}")

        self.qpos_addresses = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in robot_joint_ids], dtype=int
        )
        self.dof_addresses = np.array(
            [model.jnt_dofadr[joint_id] for joint_id in robot_joint_ids], dtype=int
        )
        name_to_mujoco = {
            name: index for index, name in enumerate(self.mujoco_joint_names)
        }
        self.policy_to_mujoco = np.array(
            [name_to_mujoco[name] for name in POLICY_JOINT_ORDER], dtype=int
        )
        self.mujoco_to_policy = np.argsort(self.policy_to_mujoco)

        self.default_position_mujoco = np.array(
            [DEFAULT_POSITION_BY_NAME.get(name, 0.0) for name in self.mujoco_joint_names],
            dtype=np.float64,
        )
        gains = [gains_for_joint(name) for name in self.mujoco_joint_names]
        self.kp = np.array([gain[0] for gain in gains], dtype=np.float64)
        self.kd = np.array([gain[1] for gain in gains], dtype=np.float64)

        joint_id_to_actuator = {
            int(model.actuator_trnid[actuator_id, 0]): actuator_id
            for actuator_id in range(model.nu)
        }
        self.actuator_ids = np.array(
            [joint_id_to_actuator[joint_id] for joint_id in robot_joint_ids], dtype=int
        )

        self.base_qpos_address = int(model.joint("floating_base_joint").qposadr[0])
        self.base_dof_address = int(model.joint("floating_base_joint").dofadr[0])
        self.bar_qpos_address = int(model.joint("object_freejoint").qposadr[0])
        self.bar_dof_address = int(model.joint("object_freejoint").dofadr[0])

        self.policy = torch.jit.load(
            str(policy_path.resolve()), map_location=self.device
        ).to(self.device).eval()
        self.previous_action = np.zeros(N_ACTIONS, dtype=np.float32)
        self.target_position_mujoco = self.default_position_mujoco.copy()
        self.observation_history: deque[np.ndarray] = deque(
            maxlen=OBS_HISTORY_LENGTH
        )
        self.policy_step_count = 0

        self.reset_simulation()
        self._validate_policy()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA")
        return torch.device(requested)

    def reset_simulation(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[
            self.base_qpos_address : self.base_qpos_address + 7
        ] = np.concatenate((ROBOT_INITIAL_POSITION, [1.0, 0.0, 0.0, 0.0]))
        self.data.qpos[self.qpos_addresses] = self.default_position_mujoco
        self.data.qpos[
            self.bar_qpos_address : self.bar_qpos_address + 7
        ] = np.concatenate((BAR_INITIAL_CENTER, [1.0, 0.0, 0.0, 0.0]))
        self.data.qvel.fill(0.0)
        self.data.ctrl.fill(0.0)
        self.data.qfrc_applied.fill(0.0)
        mujoco.mj_forward(self.model, self.data)

        self.previous_action.fill(0.0)
        self.target_position_mujoco[:] = self.default_position_mujoco
        self.observation_history.clear()
        self.policy_step_count = 0
        self.verify_centered_reset()

    def verify_centered_reset(self) -> None:
        center = self.data.qpos[
            self.bar_qpos_address : self.bar_qpos_address + 3
        ]
        robot_root = self.data.qpos[
            self.base_qpos_address : self.base_qpos_address + 3
        ]
        expected_relative = BAR_INITIAL_CENTER - ROBOT_INITIAL_POSITION
        if not np.allclose(center - robot_root, expected_relative, atol=1e-12):
            raise RuntimeError("Centered bar-to-robot reset offset is inconsistent")
        negative_endpoint, controller_point, positive_endpoint = centered_bar_sites(
            self.model, self.data
        )
        if not np.allclose(
            negative_endpoint,
            BAR_INITIAL_CENTER + BAR_NEGATIVE_Y_ENDPOINT_OFFSET,
            atol=1e-12,
        ):
            raise RuntimeError("Negative-Y bar endpoint is inconsistent")
        if not np.allclose(
            positive_endpoint,
            BAR_INITIAL_CENTER + BAR_POSITIVE_Y_ENDPOINT_OFFSET,
            atol=1e-12,
        ):
            raise RuntimeError("Positive-Y bar endpoint is inconsistent")
        if not np.allclose(controller_point, BAR_INITIAL_CENTER, atol=1e-12):
            raise RuntimeError("Controller point is not at the bar center")

    def _observation_frame(self) -> np.ndarray:
        joint_position_mujoco = self.data.qpos[self.qpos_addresses]
        joint_velocity_mujoco = self.data.qvel[self.dof_addresses]
        joint_position_policy = (
            joint_position_mujoco - self.default_position_mujoco
        )[self.policy_to_mujoco]
        joint_velocity_policy = joint_velocity_mujoco[self.policy_to_mujoco]

        base_quaternion = self.data.qpos[
            self.base_qpos_address + 3 : self.base_qpos_address + 7
        ]
        base_angular_velocity = self.data.qvel[
            self.base_dof_address + 3 : self.base_dof_address + 6
        ]
        projected_gravity = rotate_inverse(
            base_quaternion, np.array([0.0, 0.0, -1.0])
        )

        frame = np.concatenate(
            (
                np.array(
                    [0.0, 0.0, 0.0, MASKED_STUDENT_HEIGHT_COMMAND],
                    dtype=np.float32,
                ),
                LEFT_PALM_QUATERNION,
                RIGHT_PALM_QUATERNION,
                LEFT_PALM_POSITION,
                RIGHT_PALM_POSITION,
                (base_angular_velocity * 0.25).astype(np.float32),
                projected_gravity.astype(np.float32),
                joint_position_policy.astype(np.float32),
                (joint_velocity_policy * 0.05).astype(np.float32),
                self.previous_action,
            )
        )
        if frame.shape != (OBS_FRAME_SIZE,):
            raise RuntimeError(f"Unexpected student frame shape {frame.shape}")
        return frame

    def _stacked_observation(self) -> torch.Tensor:
        frame = self._observation_frame()
        if not self.observation_history:
            self.observation_history.extend(
                frame.copy() for _ in range(OBS_HISTORY_LENGTH)
            )
        else:
            self.observation_history.append(frame)
        stacked = np.clip(
            np.concatenate(tuple(self.observation_history)), -OBS_CLIP, OBS_CLIP
        )
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer_action(self) -> np.ndarray:
        output = self.policy(self._stacked_observation())
        if isinstance(output, (tuple, list)):
            output = output[0]
        action = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if action.shape != (N_ACTIONS,):
            raise RuntimeError(f"Expected 29 policy actions, received {action.shape}")
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Policy produced a non-finite action")
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        action_mujoco = action[self.mujoco_to_policy]
        self.target_position_mujoco = (
            self.default_position_mujoco + ACTION_SCALE * action_mujoco
        )
        self.previous_action = action.copy()
        self.policy_step_count += 1
        return action

    def apply_robot_pd(self) -> None:
        joint_position = self.data.qpos[self.qpos_addresses]
        joint_velocity = self.data.qvel[self.dof_addresses]
        torque = (
            self.kp * (self.target_position_mujoco - joint_position)
            - self.kd * joint_velocity
        )
        self.data.ctrl[self.actuator_ids] = torque

    def _validate_policy(self) -> None:
        action = self.infer_action()
        if action.shape != (N_ACTIONS,):
            raise RuntimeError("Policy validation failed")
        self.previous_action.fill(0.0)
        self.target_position_mujoco[:] = self.default_position_mujoco
        self.observation_history.clear()
        self.policy_step_count = 0

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[self.base_qpos_address + 2])

    @property
    def base_position(self) -> np.ndarray:
        return self.data.qpos[
            self.base_qpos_address : self.base_qpos_address + 3
        ].copy()


def parser():
    result = controller_parser()
    result.description = (
        "Evaluate a COLA student with G1, a world-Y-centered bar, and "
        "center-point control at 1 kHz."
    )
    result.set_defaults(
        model=DEFAULT_MODEL,
        height=0.75,
        duration=0.0,
    )
    model_action = next(action for action in result._actions if action.dest == "model")
    model_action.help = (
        "Centered bar model: model/centered_bar_scene.xml (free, default) or "
        "model/centered_fixed_bar_scene.xml (left weld + right point)."
    )
    result.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    result.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or a CUDA device such as cuda:0.",
    )
    return result


def centered_bar_sites(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the negative-Y endpoint, controller point, and positive-Y endpoint."""

    negative_endpoint = data.site_xpos[model.site("negative_y_endpoint").id].copy()
    controller_point = data.site_xpos[model.site("controller_point").id].copy()
    positive_endpoint = data.site_xpos[model.site("positive_y_endpoint").id].copy()
    return negative_endpoint, controller_point, positive_endpoint


def centered_bar_row(
    sample,
    negative_endpoint: np.ndarray,
    positive_endpoint: np.ndarray,
) -> dict[str, float]:
    bar_axis = positive_endpoint - negative_endpoint
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
        "controller_point_x": sample.endpoint_position[0],
        "controller_point_y": sample.endpoint_position[1],
        "controller_point_z": sample.endpoint_position[2],
        "negative_y_endpoint_x": negative_endpoint[0],
        "negative_y_endpoint_y": negative_endpoint[1],
        "negative_y_endpoint_z": negative_endpoint[2],
        "positive_y_endpoint_x": positive_endpoint[0],
        "positive_y_endpoint_y": positive_endpoint[1],
        "positive_y_endpoint_z": positive_endpoint[2],
        "bar_axis_world_x": bar_axis[0],
        "bar_axis_world_y": bar_axis[1],
        "bar_axis_world_z": bar_axis[2],
        "bar_axis_norm": np.linalg.norm(bar_axis),
    }


def bar_hand_contact_counts(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[int, int]:
    bar_geom_id = model.geom("carried_bar").id
    left_body_id = model.body("left_wrist_yaw_link").id
    right_body_id = model.body("right_wrist_yaw_link").id
    left_contacts = 0
    right_contacts = 0
    for contact in data.contact[: data.ncon]:
        if contact.geom1 == bar_geom_id:
            other_geom = contact.geom2
        elif contact.geom2 == bar_geom_id:
            other_geom = contact.geom1
        else:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        left_contacts += int(other_body == left_body_id)
        right_contacts += int(other_body == right_body_id)
    return left_contacts, right_contacts


def is_centered_fixed_bar(model: mujoco.MjModel) -> bool:
    return (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "left_bar_fixed_link",
        )
        >= 0
    )


def simulate_physics_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    evaluator: StudentEvaluator,
    controller: EndpointForceController,
    physics_step: int,
):
    if physics_step % POLICY_DECIMATION == 0:
        evaluator.infer_action()
    data.qfrc_applied.fill(0.0)
    evaluator.apply_robot_pd()
    sample = controller.compute(model, data)
    controller.apply(model, data, sample)
    mujoco.mj_step(model, data)
    return sample


def write_csv_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} physics samples to {path}")


def main() -> int:
    args = parser().parse_args()
    if args.headless and args.duration <= 0.0:
        raise SystemExit("--headless requires a positive --duration")
    if not args.model.is_file():
        raise SystemExit(f"MuJoCo model not found: {args.model}")
    if not args.policy.is_file():
        raise SystemExit(f"Student policy not found: {args.policy}")

    model, data = load_model(str(args.model.resolve()))
    if not np.isclose(model.opt.timestep, SIM_DT, atol=1e-12):
        raise RuntimeError(
            f"Expected a {SIM_DT:g} s physics timestep, got {model.opt.timestep:g}"
        )
    evaluator = StudentEvaluator(model, data, args.policy, args.device)
    controller = EndpointForceController(
        model,
        make_config(args, dt=SIM_DT),
        site_name="controller_point",
    )
    controller.set_targets(height=args.height, velocity_xy=(args.vx, args.vy))
    controller.reset(model, data)
    fixed_bar = is_centered_fixed_bar(model)

    rows: list[dict[str, float]] = []
    step_applied = False
    last_report_time = -1.0
    physics_step = 0

    def reset_all() -> None:
        nonlocal physics_step, step_applied, last_report_time
        evaluator.reset_simulation()
        controller.reset(model, data)
        physics_step = 0
        step_applied = False
        last_report_time = -1.0
        print("Reset robot, bar, student history, and endpoint controller.")

    negative_endpoint, controller_point, positive_endpoint = centered_bar_sites(
        model, data
    )
    print(f"Policy: {args.policy.resolve()}")
    print(f"Torch device: {evaluator.device}")
    print("MuJoCo/controller: 1000 Hz; student policy: 50 Hz (decimation 20)")
    if fixed_bar:
        print("Robot: G1 29-DOF with simple wrist-to-bar connector links")
    else:
        print("Robot: G1 29-DOF with fixed finger links and actuated wrists")
    print(f"Initial bar center: {np.round(BAR_INITIAL_CENTER, 3)}")
    print(
        "Initial bar points: "
        f"negative-Y={np.round(negative_endpoint, 3)}, "
        f"center/controller={np.round(controller_point, 3)}, "
        f"positive-Y={np.round(positive_endpoint, 3)}"
    )
    print(
        "Controller defaults: requested height is the mean of [0.55, 0.95] m; "
        "requested horizontal velocity is (0, 0) m/s"
    )
    print("Force arrows: blue=height PD, magenta=horizontal velocity PID")
    if fixed_bar:
        print("Bar attachment: left weld plus right point constraint.")
        print("Detailed palm and finger meshes are absent in this fixed model.")
    else:
        print("The bar is contact-held; no weld or equality constraint is used.")
    print("Controller force and feedback are both at the bar center.")
    print("No anti-sliding end plates are present.")

    def record(sample, periodic_report: bool) -> None:
        nonlocal last_report_time
        negative_end, _, positive_end = centered_bar_sites(model, data)
        left_contacts, right_contacts = bar_hand_contact_counts(model, data)
        if args.csv is not None:
            result = centered_bar_row(sample, negative_end, positive_end)
            result.update(
                {
                    "base_x": evaluator.base_position[0],
                    "base_y": evaluator.base_position[1],
                    "base_z": evaluator.base_position[2],
                    "left_hand_contacts": left_contacts,
                    "right_hand_contacts": right_contacts,
                    "policy_step": evaluator.policy_step_count,
                }
            )
            rows.append(result)
        if periodic_report and sample.time - last_report_time >= 0.5:
            last_report_time = sample.time
            attachment_status = (
                "attachments=(left-weld,right-point)"
                if fixed_bar
                else f"hand_contacts=({left_contacts},{right_contacts})"
            )
            print(
                f"t={sample.time:6.2f}  base_z={evaluator.base_height:+.3f}  "
                f"bar_h={sample.endpoint_position[2]:+.3f}/"
                f"{sample.height_reference:+.3f}  "
                f"vxy=({sample.endpoint_velocity[0]:+.3f},"
                f"{sample.endpoint_velocity[1]:+.3f})  "
                f"F=({sample.force_world[0]:+.1f},"
                f"{sample.force_world[1]:+.1f},"
                f"{sample.force_world[2]:+.1f})  "
                f"{attachment_status}"
            )

    def run_one_step(periodic_report: bool):
        nonlocal physics_step, step_applied
        step_applied = maybe_apply_step(
            args, controller, step_applied, float(data.time)
        )
        sample = simulate_physics_step(
            model, data, evaluator, controller, physics_step
        )
        physics_step += 1
        record(sample, periodic_report)
        return sample

    if args.headless:
        while data.time < args.duration:
            run_one_step(periodic_report=True)
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        ) as viewer:
            viewer_options = ViewerDisplayOptions()
            terminal = TerminalCommandInput(
                controller,
                reset_all,
                viewer_options.set_bar_vector,
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
                sample = run_one_step(periodic_report=False)
                with viewer.lock():
                    update_force_arrows(
                        viewer.user_scn,
                        sample,
                        scale=args.force_arrow_scale,
                        max_length=args.force_arrow_max_length,
                    )
                    if viewer_options.show_bar_vector:
                        negative_end, _, positive_end = centered_bar_sites(model, data)
                        update_bar_vector_arrow(
                            viewer.user_scn,
                            negative_end,
                            positive_end,
                            radius=args.bar_vector_radius,
                        )
                viewer.sync()
                if args.realtime_factor > 0.0:
                    target_wall_dt = model.opt.timestep / args.realtime_factor
                    remaining = target_wall_dt - (time.perf_counter() - step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
            print(f"Wall time: {time.perf_counter() - wall_start:.2f} s")

            # MuJoCo 3.10 can fault while destroying a viewer that owns a
            # closed-loop equality model. Simulation and rendering are already
            # complete here, so bypass library teardown for the fixed variant.
            if fixed_bar:
                if args.csv is not None:
                    write_csv_rows(args.csv, rows)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)

    if args.csv is not None:
        write_csv_rows(args.csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
