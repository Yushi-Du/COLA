# COLA browser-local MuJoCo demo

This is the static, public-site architecture used by the COLA project-page
demo. MuJoCo physics and ONNX policy inference both run in the visitor's web
browser. No Python process, local COLA checkout, Isaac Sim installation, or
remote simulation server is required after the static files have loaded.

The implementation preserves the validated native sim2sim contract:

- centered fixed bar with a left weld and right point constraint;
- the same 29-DOF G1 model and visual/collision meshes;
- 1000 Hz MuJoCo physics, 400 Hz external controllers, and a 50 Hz student;
- 25 frames of 111 observations (`2775` ONNX inputs) and 29 policy actions;
- exact nominal SMOKV3SP actuator Kp/Kd values;
- center-point height PD (`800/35`, `300 N` limit), planar velocity PID
  (`30/60/0.1`, `100 N` limit), and projected yaw PD (`40/4`, `10 N·m` limit);
- robot-local planar commands, a `[0.55, 0.85] m` height envelope, and
  `[-0.5, 0.5] m/s` planar envelopes.

The TorchScript student is exported by `../tools/export_policy_onnx.py`. Its
ONNX parity report is stored beside
`assets/policy-loop-static-three-jitter-phase3-model-5000-2433dac7.onnx`.

## Local run

From the website repository root:

```bash
python demo/wasm/tools/serve.py
```

Open <http://127.0.0.1:8766/demo/wasm/>. The helper server supplies the
cross-origin-isolation headers needed by the official multithreaded MuJoCo
WASM build.

On static GitHub Pages, `coi-serviceworker.js` injects equivalent headers. The
first visit can reload once after registering that service worker. GitHub Pages
is HTTPS, which satisfies the secure-context requirement.

## Verification

Export and verify the ONNX policy:

```bash
conda run -n env_isaaclab python demo/tools/export_policy_onnx.py
```

The deterministic browser smoke page is
`/demo/wasm/tests/browser-smoke.html?steps=5000`. It disables the real-time
animation loop, advances exactly the requested number of physics steps, and
writes its result and final state to the document body for headless testing.

The main page also accepts `?smoke=1`; this advances 240 steps and freezes a
full-UI review frame.

## Deployment size and support

The static payload is about 47 MB before HTTP compression, primarily robot STL
meshes, MuJoCo WASM, ONNX Runtime WASM, and the policy. Browsers cache these
assets after the first load. The initial release targets current desktop
Chrome/Edge and Firefox, matching the stated project scope; mobile layout and
mobile performance are not yet treated as release requirements.
