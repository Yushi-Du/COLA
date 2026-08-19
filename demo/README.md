# COLA interactive MuJoCo demos

The project-page **Demo** button now targets `wasm/`, a static browser-local
port documented in [`wasm/README.md`](wasm/README.md). Its MuJoCo physics and
ONNX policy inference run on the visitor's computer without a Python backend.

The files directly in this directory (`server.py`, `index.html`, `demo.js`, and
`demo.css`) preserve the earlier native Python/TorchScript prototype as a
validation reference. They have not been removed or overwritten.

The browser is the control/display surface. Native MuJoCo and the exported
TorchScript student continue to run in Python, so this prototype uses the same:

- centered fixed bar and left-weld/right-point attachment topology;
- 29-DOF G1 model, student observation history, policy inference and SMOKV3SP
  robot Kp/Kd;
- 1000 Hz physics, 400 Hz external controllers and 50 Hz policy;
- center-point height PD, horizontal-velocity PID and projected world-yaw PD;
- `[0.55, 0.85] m` height and `[-0.50, 0.50] m/s` planar command limits.

The viewer fills the browser window and streams a native 3840×2160 MuJoCo
render. Its controls are kept in one compact translucent dock over the scene.
For this local viewer only, the ground plane's visual X/Y half-sizes are set to
zero so MuJoCo renders it as infinite; the source sim2sim XML and contact model
remain unchanged.

The physical carried bar is also retained unchanged but rendered transparent.
Its website-only visual is a short, thick, semi-transparent gray,
non-colliding rectangular bar along the line between the robot's two hands. It
spans both complete hand meshes, so the retained robot hands appear embedded
inside the bar. The previous forward-facing box and virtual human hands are
removed. Controller-force,
velocity, torque, and bar-vector debug overlays start disabled; `V` toggles all
of them when debugging is needed.

## Run

```bash
conda activate base
cd /path/to/COLA
python demo/server.py
```

Then open the project page at <http://127.0.0.1:8765/> and select **Demo**, or
open <http://127.0.0.1:8765/demo/> directly. MuJoCo is initialized lazily after
the demo route is requested, so the browser can display the loading screen
while the model, controllers, and student policy are prepared.

Every page load restores only the command targets to their website defaults:
zero planar velocity, `0.70 m` height, and `-90 deg` target-vector yaw. The
robot and object physical state are not reset by opening or refreshing the
page.

Override the policy or port when needed:

```bash
python demo/server.py --policy /absolute/path/to/policy_student_model_N.jit --port 8765
```

The server binds to `127.0.0.1` by default and is not exposed to the network.
Stop it with `Ctrl+C`.

## Default policy provenance

The native and browser-local demos are pinned to the replacement static-three
jitter phase-3 checkpoint supplied for the website:

- checkpoint: `loop_static_three_jitter_phase3_model_5000/model_5000.pt`,
  iteration 5000;
- raw checkpoint SHA-256:
  `2433dac7f0975d91564f27718c5e844f767d652938d95778cb6c2d7e03fbcab8`;
- exported student:
  `policy_student_loop_static_three_jitter_phase3_model_5000.jit`;
- TorchScript SHA-256:
  `7a32546b2931734d229af47f960b336396d0d09e3ba3144eefe83b9afbae66ed`;
- browser ONNX SHA-256:
  `fdeb8320fd67ed15c46fdf0d3a5503c16c61c22e12f939fdd62f0a476aca9ae5`.

The raw checkpoint remains outside this website checkout. A byte-identical copy
of the exported policy is stored in `demo/models/`, alongside the website-local
sim2sim runtime under `demo/sim2sim_runtime/`.

## Controls

The right-side dock combines continuous controls with the original keyboard
shortcuts:

- `W/S`: robot-local forward/backward velocity ±0.05 m/s;
- `A/D`: robot-local left/right velocity ±0.05 m/s;
- `I/K`: target height ±0.01 m;
- `J/L`: target-vector world yaw ±5 degrees;
- `C`: reset target-vector yaw to −90 degrees;
- `X`: zero planar target velocity;
- `R`: reset robot, bar, observation history and controllers.

Robot-local forward and lateral velocity share a two-dimensional coordinate
pad whose center is zero; world target height has a continuous slider. The
target-vector control is a white 3D coordinate view: the solid gray cuboid is
the measured bar-vector yaw projection and the translucent cyan cuboid is its
target. Dragging in that view directly changes target yaw; `J/L/C` remain
available for precise keyboard adjustment.

The page additionally exposes pause/resume, camera follow, drag-to-orbit and
scroll-to-zoom controls. External controllers remain enabled and debug overlays
remain hidden in the website demo.

The lightweight left-side monitor uses three bounded, three-second history
plots sampled at about 25 Hz. Lateral velocity force is shown as `|Fy|`, while
height force and world-Z torque retain their signs. Each plot has an independent
arrow switch. These switches reuse the native MuJoCo velocity-force,
height-force, and yaw-torque arrows with their original geometry and colors;
they do not disable the underlying controller.

The planar target is rotated from the robot's heading frame into the world
frame at each 400 Hz controller evaluation. The existing world-frame PID and
world-frame applied forces are otherwise unchanged.

## Legacy architecture note

The native backend remains useful for side-by-side debugging. The public static
port is now implemented under `wasm/`; the original local sim2sim pipelines are
still not imported or modified by either website demo.
