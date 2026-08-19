from __future__ import annotations

import json
from pathlib import Path
import unittest


WASM_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = WASM_ROOT.parent
REPOSITORY_ROOT = DEMO_ROOT.parent


class BrowserDemoStaticContractTests(unittest.TestCase):
    def test_project_demo_button_targets_static_wasm_demo(self) -> None:
        homepage = (REPOSITORY_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="demo/wasm/"', homepage)
        self.assertIn('<link rel="icon" type="image/png" href="static/images/favicon-cola.png?v=2">', homepage)
        self.assertTrue((REPOSITORY_ROOT / "static" / "images" / "favicon-cola.png").is_file())

    def test_homepage_waits_for_every_video(self) -> None:
        homepage = (REPOSITORY_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPOSITORY_ROOT / "static" / "js" / "index.js").read_text(encoding="utf-8")
        styles = (REPOSITORY_ROOT / "static" / "css" / "index.css").read_text(encoding="utf-8")
        self.assertEqual(homepage.count("<video"), 7)
        self.assertEqual(homepage.count('preload="auto"'), 7)
        self.assertNotIn("raw.githubusercontent.com/Yushi-Du/COLA/main/static/videos/", homepage)
        video_names = (
            "Video_Release_No_Method_2.mp4",
            "Short_clip_id1_masked.mp4",
            "Short_clip_id2.mp4",
            "Short_clip_id3.mp4",
            "Short_clip_id4.mp4",
            "Short_clip_id8.mp4",
            "Short_clip_id9.mp4",
        )
        for video_name in video_names:
            self.assertIn(f'src="static/videos/{video_name}"', homepage)
            self.assertTrue((REPOSITORY_ROOT / "static" / "videos" / video_name).is_file())
        self.assertIn('class="homepage-loader"', homepage)
        self.assertIn("Loading contents 0 / 7", homepage)
        self.assertIn("Accepted by ICRA 2026", homepage)
        self.assertIn("Loading contents", script)
        self.assertNotIn("Loading videos", homepage + script)
        self.assertIn("HTMLMediaElement.HAVE_CURRENT_DATA", script)
        self.assertIn("loadeddata", script)
        self.assertIn("Promise.all(videos.map", script)
        self.assertIn(".homepage-loader.is-complete", styles)

    def test_homepage_visual_refresh_preserves_reduced_motion(self) -> None:
        homepage = (REPOSITORY_ROOT / "index.html").read_text(encoding="utf-8")
        script = (REPOSITORY_ROOT / "static" / "js" / "index.js").read_text(encoding="utf-8")
        styles = (REPOSITORY_ROOT / "static" / "css" / "index.css").read_text(encoding="utf-8")
        self.assertIn("IntersectionObserver", script)
        self.assertIn("project-reveal", script)
        self.assertIn("startRevealAnimations", script)
        self.assertIn("--cola-ink: #172033", styles)
        self.assertIn("radial-gradient(circle, rgba(69, 84, 105", styles)
        self.assertIn("font-family: 'Castoro', Georgia, serif", styles)
        self.assertIn(".page-ready .project-reveal.is-visible", styles)
        self.assertIn("@media (prefers-reduced-motion: reduce)", styles)
        self.assertIn('src="demo/wasm/assets/cola-icon-transparent.png"', homepage)
        self.assertIn('src="static/images/pipeline-transparent.png"', homepage)
        self.assertTrue((REPOSITORY_ROOT / "static" / "images" / "pipeline-transparent.png").is_file())
        self.assertIn('class="pipeline-figure"', homepage)
        self.assertIn("demo-button", homepage)
        self.assertIn(".h2-text {\n  display: inline;", styles)
        self.assertIn("border: 0;\n  border-radius: 0;\n  background: transparent;", styles)
        self.assertIn(".content-section .pipeline-figure", styles)
        self.assertIn("@keyframes demo-color-flow", styles)
        self.assertIn("@keyframes demo-soft-glow", styles)
        self.assertIn("font-size: 0.98rem;", styles)
        self.assertGreaterEqual(styles.count("font-kerning: none;"), 3)
        self.assertGreaterEqual(styles.count("font-variant-ligatures: none;"), 3)
        self.assertGreaterEqual(styles.count("letter-spacing: 0.01em;"), 2)
        self.assertIn("letter-spacing: 0.008em;", styles)
        self.assertIn("margin-right: 0.02em;", styles)

    def test_page_uses_canvas_and_local_es_modules(self) -> None:
        html = (WASM_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('<canvas id="simulation-stream"', html)
        self.assertIn('href="../../static/images/favicon-cola.png?v=2"', html)
        self.assertNotIn('/api/stream', html)
        self.assertIn('src="coi-serviceworker.js?v=model5000-2433dac7"', html)
        self.assertIn('type="module" src="demo.js?v=model5000-2433dac7"', html)
        self.assertIn('"three": "./vendor/three/three.module.min.js"', html)

    def test_browser_runtime_has_no_simulation_backend_requests(self) -> None:
        ui = (WASM_ROOT / "demo.js").read_text(encoding="utf-8")
        runtime = (WASM_ROOT / "runtime.js").read_text(encoding="utf-8")
        self.assertNotIn("fetch(", ui)
        self.assertNotIn("/api/", ui)
        self.assertIn("invokeRuntime", ui)
        self.assertIn("new ColaBrowserRuntime", ui)
        self.assertIn("window.crossOriginIsolated", ui)
        self.assertIn("mujoco.mj_step", runtime)
        self.assertIn("new StudentPolicy", runtime)

    def test_pose_panel_uses_the_three_camera_basis(self) -> None:
        source = (WASM_ROOT / "demo.js").read_text(encoding="utf-8")
        self.assertIn("x: Math.cos(azimuth),\n    y: Math.sin(azimuth)", source)
        self.assertIn("x: Math.sin(azimuth) * Math.sin(elevation)", source)
        self.assertIn("y: -Math.cos(azimuth) * Math.sin(elevation)", source)

    def test_loading_panel_has_cola_identity_and_motion(self) -> None:
        html = (WASM_ROOT / "index.html").read_text(encoding="utf-8")
        css = (WASM_ROOT / "demo.css").read_text(encoding="utf-8")
        ui = (WASM_ROOT / "demo.js").read_text(encoding="utf-8")
        self.assertLess(html.index("Preparing</span>"), html.index('class="cola-icon"'))
        self.assertLess(html.index('class="cola-icon"'), html.index("COLA in Mujoco</span>"))
        self.assertIn('class="cola-icon"', html)
        self.assertIn('src="assets/cola-icon-transparent.png"', html)
        self.assertTrue((WASM_ROOT / "assets" / "cola-icon-transparent.png").is_file())
        self.assertIn("@keyframes cola-jitter", css)
        self.assertIn("not(.is-error) .cola-icon", css)
        self.assertIn(".loading-title { display: inline-flex; align-items: center; gap: 0; }", css)
        self.assertIn("margin-left: 0;", css)
        self.assertIn("drop-shadow(0 0 2px", css)
        self.assertIn("drop-shadow(0 0 6px", css)
        self.assertIn("'COLA in Mujoco'", ui)

    def test_rates_observation_width_and_command_limits_are_pinned(self) -> None:
        constants = (WASM_ROOT / "constants.js").read_text(encoding="utf-8")
        for contract in (
            "PHYSICS_HZ = 1000",
            "CONTROLLER_HZ = 400",
            "POLICY_HZ = 50",
            "OBS_FRAME_SIZE = 111",
            "OBS_HISTORY_LENGTH = 25",
            "HEIGHT_LIMITS = Object.freeze([0.55, 0.85])",
            "VELOCITY_X_LIMITS = Object.freeze([-0.5, 0.5])",
            "VELOCITY_Y_LIMITS = Object.freeze([-0.5, 0.5])",
        ):
            self.assertIn(contract, constants)

    def test_controller_gains_and_limits_match_native_demo(self) -> None:
        controllers = (WASM_ROOT / "controllers.js").read_text(encoding="utf-8")
        for contract in (
            "heightKp: 800",
            "heightKd: 35",
            "heightForceLimit: 300",
            "velocityKp: 30",
            "velocityKi: 60",
            "velocityKd: 0.1",
            "horizontalForceLimit: 100",
            "kp: 40",
            "kd: 4",
            "torqueLimit: 10",
        ):
            self.assertIn(contract, controllers)

    def test_policy_export_passed_numerical_parity(self) -> None:
        policy_name = "policy-loop-static-three-jitter-phase3-model-5000-2433dac7.onnx"
        policy = WASM_ROOT / "assets" / policy_name
        report_path = policy.with_suffix(".parity.json")
        self.assertGreater(policy.stat().st_size, 1_000_000)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["checkpoint"], "model_5000.pt")
        self.assertEqual(
            report["checkpoint_sha256"],
            "2433dac7f0975d91564f27718c5e844f767d652938d95778cb6c2d7e03fbcab8",
        )
        self.assertEqual(
            report["source_sha256"],
            "7a32546b2931734d229af47f960b336396d0d09e3ba3144eefe83b9afbae66ed",
        )
        self.assertEqual(
            report["onnx_sha256"],
            "fdeb8320fd67ed15c46fdf0d3a5503c16c61c22e12f939fdd62f0a476aca9ae5",
        )
        self.assertEqual(report["input_shape"], [32, 2775])
        self.assertEqual(report["output_shape"], [32, 29])
        self.assertLessEqual(report["maximum_absolute_error"], 2.5e-5)
        runtime = (WASM_ROOT / "runtime.js").read_text(encoding="utf-8")
        self.assertIn(policy_name, runtime)

    def test_all_model_meshes_are_website_local(self) -> None:
        robot_xml = (
            WASM_ROOT / "assets" / "model" / "g1" / "g1_29dof_centered_fixed_bar.xml"
        ).read_text(encoding="utf-8")
        mesh_names = [part.split('"', 1)[0] for part in robot_xml.split('file="')[1:]]
        self.assertGreater(len(mesh_names), 20)
        for mesh_name in mesh_names:
            self.assertTrue((WASM_ROOT / "assets" / "model" / "g1" / mesh_name).is_file())


if __name__ == "__main__":
    unittest.main()
