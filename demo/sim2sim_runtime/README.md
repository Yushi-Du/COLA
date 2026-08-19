# Website-local sim2sim runtime

This is a minimal functional copy of the centered fixed-middle-bar MuJoCo
pipeline used by the local COLA website demo. It includes only the Python
modules, scene XML, and 33 G1 meshes required by that demo.

Copied from the sibling workspace pipelines on 2026-08-17:

- `mujoco_fixed_bar_middle_controller_sim2sim`;
- `mujoco_centered_bar_controller_sim2sim`.

Website-only differences:

- planar velocity commands are stored in the robot heading frame and rotated
  into world coordinates at each 400 Hz endpoint-controller evaluation;
- the accepted height-command range is `[0.55, 0.85] m`;
- the default policy resolves to
  `demo/models/policy_student_loop_static_three_jitter_phase3_model_5000.jit`.
- the original bar collision is transparent and a zero-collision box plus four
  visual hand meshes provide the website presentation geometry.

The original workspace pipelines are not imported at runtime and were not
modified while creating this copy.
