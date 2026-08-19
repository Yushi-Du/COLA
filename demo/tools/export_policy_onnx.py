#!/usr/bin/env python3
"""Export the browser student policy and verify ONNX numerical parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch


DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    DEMO_ROOT / "models" / "policy_student_loop_static_three_jitter_phase3_model_5000.jit"
)
DEFAULT_OUTPUT = (
    DEMO_ROOT
    / "wasm"
    / "assets"
    / "policy-loop-static-three-jitter-phase3-model-5000-2433dac7.onnx"
)
OBSERVATION_SIZE = 2775
ACTION_SIZE = 29


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional raw checkpoint used to produce the TorchScript student",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    policy = torch.jit.load(str(args.input.resolve()), map_location="cpu").eval()
    dummy = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
    with torch.inference_mode():
        output = policy(dummy)
    if tuple(output.shape) != (1, ACTION_SIZE):
        raise RuntimeError(f"Unexpected TorchScript output shape: {tuple(output.shape)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy,
        dummy,
        str(args.output),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    model = onnx.load(str(args.output))
    onnx.checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(args.seed)
    samples = rng.standard_normal((args.samples, OBSERVATION_SIZE)).astype(np.float32)
    samples = np.clip(samples, -3.0, 3.0)
    with torch.inference_mode():
        torch_output = policy(torch.from_numpy(samples)).cpu().numpy()
    onnx_output = evaluator.run(None, {"observation": samples})[0]
    absolute_error = np.abs(torch_output - onnx_output)
    np.testing.assert_allclose(
        onnx_output,
        torch_output,
        atol=args.atol,
        rtol=args.rtol,
    )

    def display_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(DEMO_ROOT))
        except ValueError:
            return resolved.name

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    report = {
        "source": display_path(args.input),
        "source_sha256": sha256(args.input),
        "output": display_path(args.output),
        "onnx_sha256": sha256(args.output),
        "opset": 17,
        "samples": args.samples,
        "input_shape": list(samples.shape),
        "output_shape": list(onnx_output.shape),
        "maximum_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "onnx_bytes": args.output.stat().st_size,
    }
    if args.checkpoint is not None:
        report["checkpoint"] = display_path(args.checkpoint)
        report["checkpoint_sha256"] = sha256(args.checkpoint)
    report_path = args.output.with_suffix(".parity.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
