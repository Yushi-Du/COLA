from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import threading
import unittest

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = DEMO_ROOT / "server.py"


class StaticDemoContractTests(unittest.TestCase):
    def test_commands_use_sliders_pose_view_and_keyboard(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        javascript = (DEMO_ROOT / "demo.js").read_text()
        for command in "wasdikjlcx":
            self.assertNotIn(f'data-command="{command}"', html)
            self.assertIn(f"'{command}'", javascript)
        self.assertNotIn('id="controllers-toggle"', html)
        self.assertNotIn('id="vectors-toggle"', html)
        self.assertNotIn("'v'", javascript)
        for control_id in (
            "velocity-track",
            "velocity-thumb",
            "height-slider",
            "target-vector-canvas",
        ):
            self.assertIn(f'id="{control_id}"', html)

    def test_frontend_uses_only_local_api_routes(self) -> None:
        javascript = (DEMO_ROOT / "demo.js").read_text()
        for route in ("/api/state", "/api/command", "/api/velocity", "/api/height", "/api/yaw", "/api/default-targets", "/api/controller-overlay", "/api/camera", "/api/camera-follow"):
            self.assertIn(route, javascript)
        self.assertNotIn("api_key", javascript.lower())

    def test_project_page_places_demo_beside_arxiv_and_code(self) -> None:
        published_index = DEMO_ROOT.parent / "index.html"
        html = published_index.read_text()
        arxiv = html.index(">arXiv<")
        code = html.index(">Code<")
        demo = html.index(">Demo<")
        self.assertLess(arxiv, code)
        self.assertLess(code, demo)
        self.assertIn('href="demo/wasm/"', html)
        self.assertNotIn("demo/server.py", html)

    def test_loading_ui_handles_the_lazy_setup_state(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        javascript = (DEMO_ROOT / "demo.js").read_text()
        self.assertIn("Preparing MuJoCo", html)
        self.assertIn("state.ready === false", javascript)
        self.assertIn("state.loading_message", javascript)
        self.assertIn("latestState?.ready !== false", javascript)

    def test_runtime_and_policy_are_website_local(self) -> None:
        server_source = SERVER_PATH.read_text()
        self.assertIn('/ "models"', server_source)
        self.assertIn('/ "sim2sim_runtime"', server_source)
        self.assertNotIn('WORKSPACE_ROOT / "mujoco_', server_source)
        self.assertTrue(
            (
                DEMO_ROOT
                / "models"
                / "policy_student_loop_static_three_jitter_phase3_model_5000.jit"
            ).is_file()
        )

    def test_viewer_is_fullscreen_and_has_no_visible_headings(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text().lower()
        css = (DEMO_ROOT / "demo.css").read_text()
        self.assertNotIn("<h1", html)
        self.assertNotIn("<h2", html)
        self.assertIn(".viewport {", css)
        self.assertIn("position: fixed;", css)
        self.assertIn("inset: 0;", css)
        self.assertIn('class="control-dock"', html)

    def test_control_dock_is_anchored_top_right(self) -> None:
        css = (DEMO_ROOT / "demo.css").read_text()
        anchor = ".control-dock {\n  right: 20px;\n  top: 18px;"
        self.assertIn(anchor, css)
        rule = css.split(anchor, 1)[1].split("}", 1)[0]
        self.assertNotIn("translateY", rule)

    def test_pose_view_has_white_3d_axes_and_two_bar_states(self) -> None:
        css = (DEMO_ROOT / "demo.css").read_text()
        javascript = (DEMO_ROOT / "demo.js").read_text()
        pose_rule = css.split(".pose-window {", 1)[1].split("}", 1)[0]
        self.assertIn("background: #fff;", pose_rule)
        for label in ("'X'", "'Y'", "'Z'"):
            self.assertIn(label, javascript)
        self.assertIn("actualYaw", javascript)
        self.assertIn("targetYaw", javascript)
        self.assertIn("drawCuboid", javascript)

    def test_velocity_plane_is_square_and_module_headers_are_consistent(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        css = (DEMO_ROOT / "demo.css").read_text()
        velocity_rule = css.split(".velocity-track {", 1)[1].split("}", 1)[0]
        self.assertIn("aspect-ratio: 1 / 1;", velocity_rule)
        for name in ("linear velocity", "height", "target-vector yaw"):
            self.assertIn(f"<strong>{name}</strong>", html)

    def test_interface_typography_is_legible(self) -> None:
        css = (DEMO_ROOT / "demo.css").read_text()
        self.assertIn('font: 13px/1 "DM Mono", monospace;', css)
        self.assertIn('font: 11px/1 "DM Mono", monospace;', css)

    def test_status_bar_does_not_display_checkpoint_name(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        javascript = (DEMO_ROOT / "demo.js").read_text()
        self.assertNotIn('id="policy-name"', html)
        self.assertNotIn("setText('#policy-name'", javascript)

    def test_left_controller_monitors_are_ordered_and_history_is_bounded(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        javascript = (DEMO_ROOT / "demo.js").read_text()
        canvas_ids = (
            "lateral-force-canvas",
            "height-force-canvas",
            "torque-history-canvas",
        )
        positions = [html.index(f'id="{canvas_id}"') for canvas_id in canvas_ids]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("CONTROLLER_HISTORY_SECONDS = 3", javascript)
        self.assertIn("CONTROLLER_SAMPLE_INTERVAL_SECONDS = 0.04", javascript)
        self.assertIn(".slice(-96)", javascript)
        self.assertIn("window.setTimeout(pollState, 40)", javascript)
        self.assertNotIn("drawStationaryControllerBar", javascript)
        self.assertNotIn("drawTorqueArc", javascript)
        for controller in ("velocity", "height", "torque"):
            self.assertIn(f'data-overlay="{controller}"', html)

    def test_both_panels_explain_their_purpose_and_actions_are_labeled(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        self.assertIn("Controller Response", html)
        self.assertIn("Command Interface", html)
        self.assertNotIn("live force and torque history", html)
        self.assertNotIn("set robot-local motion", html)
        for label in ("camera follow", "pause", "reset"):
            self.assertIn(f">{label}<", html)

    def test_controller_response_panel_is_preserved_but_hidden_by_default(self) -> None:
        html = (DEMO_ROOT / "index.html").read_text()
        self.assertIn(
            'class="controller-visuals" aria-label="Controller visualizations" hidden',
            html,
        )
        for canvas_id in (
            "lateral-force-canvas",
            "height-force-canvas",
            "torque-history-canvas",
        ):
            self.assertIn(f'id="{canvas_id}"', html)


def load_server_module():
    spec = importlib.util.spec_from_file_location("cola_demo_server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load demo server")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SimulationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = load_server_module()
        if not cls.server.DEFAULT_POLICY.is_file():
            raise unittest.SkipTest(f"Policy is not available: {cls.server.DEFAULT_POLICY}")
        cls.simulation = cls.server.ColaSimulation(
            cls.server.DEFAULT_POLICY,
            device="cpu",
            controllers_enabled=True,
            enable_rendering=False,
        )

    def test_simulation_manager_starts_lazily(self) -> None:
        release_factory = threading.Event()
        factory_started = threading.Event()

        def blocked_factory():
            factory_started.set()
            release_factory.wait(timeout=2.0)
            raise RuntimeError("intentional test stop")

        manager = self.server.SimulationManager(blocked_factory)
        self.assertIsNone(manager.initializer)
        self.assertIsNone(manager.simulation)
        manager.start()
        self.assertTrue(factory_started.wait(timeout=1.0))
        state = manager.state()
        self.assertFalse(state["ready"])
        self.assertIn("Loading the model", state["loading_message"])
        release_factory.set()
        manager.stop()

    def test_exact_rates_topology_and_checkpoint(self) -> None:
        state = self.simulation.state()
        self.assertEqual(state["rates"], {"physics_hz": 1000.0, "controller_hz": 400.0, "policy_hz": 50.0})
        self.assertEqual(
            state["checkpoint"],
            "policy_student_loop_static_three_jitter_phase3_model_5000.jit",
        )
        self.assertEqual(
            (self.simulation.render_width, self.simulation.render_height),
            (3840, 2160),
        )
        self.assertGreaterEqual(self.simulation.model.vis.global_.offwidth, 3840)
        self.assertGreaterEqual(self.simulation.model.vis.global_.offheight, 2160)
        ground_id = self.simulation.model.geom("ground").id
        self.assertEqual(tuple(self.simulation.model.geom_size[ground_id, :2]), (0.0, 0.0))
        self.assertEqual(self.simulation.state()["limits"]["height"], [0.55, 0.85])
        self.assertEqual(self.simulation.state()["command_frame"], "robot_heading")
        self.assertEqual(
            self.simulation.state()["camera"],
            {"azimuth": 138.0, "elevation": -13.0},
        )
        self.assertIn("left_bar_fixed_link", [self.simulation.model.eq(i).name for i in range(self.simulation.model.neq)])
        self.assertIn("right_bar_fixed_link", [self.simulation.model.eq(i).name for i in range(self.simulation.model.neq)])

    def test_commands_are_clamped_to_deployment_envelope(self) -> None:
        for _ in range(100):
            self.simulation.command("k")
            self.simulation.command("w")
            self.simulation.command("a")
        state = self.simulation.state()
        self.assertEqual(state["target"]["height"], 0.55)
        self.assertEqual(state["target"]["vx"], 0.5)
        self.assertEqual(state["target"]["vy"], 0.5)

    def test_direct_velocity_target_is_exact_and_clamped(self) -> None:
        state = self.simulation.set_velocity_target(vx=0.231, vy=-0.174)
        self.assertEqual(state["target"]["vx"], 0.231)
        self.assertEqual(state["target"]["vy"], -0.174)
        state = self.simulation.set_velocity_target(vx=3.0, vy=-3.0)
        self.assertEqual(state["target"]["vx"], 0.5)
        self.assertEqual(state["target"]["vy"], -0.5)
        with self.assertRaises(ValueError):
            self.simulation.set_velocity_target(vx=float("nan"), vy=0.0)

    def test_direct_height_and_yaw_targets_are_exact_and_clamped(self) -> None:
        state = self.simulation.set_height_target(height=0.731)
        self.assertEqual(state["target"]["height"], 0.731)
        state = self.simulation.set_height_target(height=4.0)
        self.assertEqual(state["target"]["height"], 0.85)
        state = self.simulation.set_yaw_target(yaw_deg=225.0)
        self.assertAlmostEqual(state["target"]["yaw_deg"], -135.0)
        with self.assertRaises(ValueError):
            self.simulation.set_height_target(height=float("inf"))
        with self.assertRaises(ValueError):
            self.simulation.set_yaw_target(yaw_deg=float("nan"))

    def test_page_initialization_restores_only_default_targets(self) -> None:
        self.simulation.set_velocity_target(vx=0.31, vy=-0.27)
        self.simulation.set_height_target(height=0.81)
        self.simulation.set_yaw_target(yaw_deg=35.0)
        time_before = self.simulation.state()["time"]
        state = self.simulation.reset_command_targets()
        self.assertEqual(
            state["target"],
            {
                "height": 0.7,
                "vx": 0.0,
                "vy": 0.0,
                "yaw_deg": -90.0,
                "vector": [math.cos(-0.5 * math.pi), -1.0, 0.0],
            },
        )
        self.assertEqual(state["time"], time_before)

    def test_controller_arrow_overlays_toggle_independently(self) -> None:
        initial = self.simulation.state()["overlays"]
        self.assertEqual(
            initial,
            {"velocity": False, "height": False, "torque": False},
        )
        state = self.simulation.toggle_controller_overlay("velocity")
        self.assertTrue(state["overlays"]["velocity"])
        self.assertFalse(state["overlays"]["height"])
        self.assertFalse(state["overlays"]["torque"])
        state = self.simulation.toggle_controller_overlay("velocity")
        self.assertEqual(
            state["overlays"],
            {"velocity": False, "height": False, "torque": False},
        )
        with self.assertRaises(ValueError):
            self.simulation.toggle_controller_overlay("unknown")

    def test_policy_and_controllers_advance_without_nonfinite_state(self) -> None:
        self.simulation.command("r")
        state = self.simulation.run_steps_for_test(80)
        self.assertGreater(state["time"], 0.0)
        self.assertTrue(state["healthy"])
        self.assertGreaterEqual(self.simulation.evaluator.policy_step_count, 4)
        for value in state["wrench_local"].values():
            self.assertTrue(math.isfinite(value))

    def test_robot_local_velocity_is_rotated_by_current_heading(self) -> None:
        controller = self.simulation.endpoint_controller
        address = controller.base_qpos_address
        original = self.simulation.data.qpos[address + 3 : address + 7].copy()
        try:
            half_angle = 0.125 * math.tau
            self.simulation.data.qpos[address + 3 : address + 7] = [
                math.cos(half_angle),
                0.0,
                0.0,
                math.sin(half_angle),
            ]
            world = controller.local_to_world_xy(
                self.simulation.data, np.array([0.2, 0.0])
            )
            np.testing.assert_allclose(world, [0.0, 0.2], atol=1e-12)
        finally:
            self.simulation.data.qpos[address + 3 : address + 7] = original

    def test_thick_visual_bar_preserves_hidden_bar_collision_and_mass(self) -> None:
        model = self.simulation.model
        bar_id = model.geom("carried_bar").id
        np.testing.assert_allclose(model.geom_size[bar_id], [0.005, 0.8, 0.005])
        self.assertEqual(int(model.geom_contype[bar_id]), 1)
        self.assertEqual(int(model.geom_conaffinity[bar_id]), 1)
        self.assertEqual(float(model.geom_rgba[bar_id, 3]), 0.0)

        visual_id = model.geom("carried_bar_visual").id
        np.testing.assert_allclose(
            model.geom_size[visual_id], [0.065, 0.205, 0.055]
        )
        np.testing.assert_allclose(model.geom_pos[visual_id], [-0.015, 0.0, 0.015])
        visual_material_id = model.geom_matid[visual_id]
        np.testing.assert_allclose(
            model.mat_rgba[visual_material_id],
            [0.46, 0.49, 0.53, 0.58],
        )
        self.assertEqual(int(model.geom_contype[visual_id]), 0)
        self.assertEqual(int(model.geom_conaffinity[visual_id]), 0)

        scene_path = (
            DEMO_ROOT
            / "sim2sim_runtime"
            / "mujoco_centered_bar_controller_sim2sim"
            / "model"
            / "centered_fixed_bar_scene.xml"
        )
        scene_text = scene_path.read_text()
        self.assertNotIn("virtual_human", scene_text)
        visual_position = model.geom_pos[visual_id]
        visual_half_size = model.geom_size[visual_id]
        for site_name in ("left_bar_attachment", "right_bar_attachment"):
            site_position = model.site_pos[model.site(site_name).id]
            self.assertTrue(
                np.all(np.abs(site_position - visual_position) <= visual_half_size)
            )

        for name in (
            "left_robot_hand_visual",
            "right_robot_hand_visual",
        ):
            geom_id = model.geom(name).id
            self.assertEqual(int(model.geom_contype[geom_id]), 0)
            self.assertEqual(int(model.geom_conaffinity[geom_id]), 0)

        carried_body_id = model.body("carried_object").id
        self.assertEqual(float(model.body_mass[carried_body_id]), 1.0)
        np.testing.assert_allclose(
            model.body_inertia[carried_body_id],
            [0.2133417, 0.0000167, 0.2133417],
        )
        self.assertFalse(self.simulation.state()["vectors_visible"])


if __name__ == "__main__":
    unittest.main()
