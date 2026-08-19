import * as ort from './vendor/ort/ort.wasm.min.mjs';

import {
  ACTION_CLIP,
  ACTION_SCALE,
  LEFT_PALM_POSITION,
  LEFT_PALM_QUATERNION,
  MASKED_STUDENT_HEIGHT_COMMAND,
  N_ACTIONS,
  OBS_CLIP,
  OBS_FRAME_SIZE,
  OBS_HISTORY_LENGTH,
  OBSERVATION_SIZE,
  RIGHT_PALM_POSITION,
  RIGHT_PALM_QUATERNION,
  clamp,
} from './constants.js';


function rotateInverse(quaternion, vector) {
  const norm = Math.hypot(...quaternion);
  const w = quaternion[0] / norm;
  const x = quaternion[1] / norm;
  const y = quaternion[2] / norm;
  const z = quaternion[3] / norm;
  const vx = vector[0];
  const vy = vector[1];
  const vz = vector[2];
  const dot = x * vx + y * vy + z * vz;
  const crossX = y * vz - z * vy;
  const crossY = z * vx - x * vz;
  const crossZ = x * vy - y * vx;
  const scale = 2 * w * w - 1;
  return [
    vx * scale - 2 * w * crossX + 2 * x * dot,
    vy * scale - 2 * w * crossY + 2 * y * dot,
    vz * scale - 2 * w * crossZ + 2 * z * dot,
  ];
}


export async function createPolicySession(policyUrl) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = {
    mjs: new URL('./vendor/ort/ort-wasm-simd-threaded.mjs', import.meta.url),
    wasm: new URL('./vendor/ort/ort-wasm-simd-threaded.wasm', import.meta.url),
  };
  const session = await ort.InferenceSession.create(policyUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  if (!session.inputNames.includes('observation')) {
    throw new Error(`ONNX input mismatch: ${session.inputNames.join(', ')}`);
  }
  return session;
}


export class StudentPolicy {
  constructor(model, data, mapping) {
    this.model = model;
    this.data = data;
    this.mapping = mapping;
    this.session = null;
    this.previousAction = new Float32Array(N_ACTIONS);
    this.targetPositionMujoco = Float64Array.from(mapping.defaultPositionMujoco);
    this.observationHistory = [];
    this.policyStepCount = 0;
  }

  initialize(session) {
    this.session = session;
  }

  reset() {
    this.previousAction.fill(0);
    this.targetPositionMujoco.set(this.mapping.defaultPositionMujoco);
    this.observationHistory = [];
    this.policyStepCount = 0;
  }

  observationFrame() {
    const { qpos, qvel } = this.data;
    const {
      qposAddresses,
      dofAddresses,
      defaultPositionMujoco,
      policyToMujoco,
      baseQposAddress,
      baseDofAddress,
    } = this.mapping;
    const frame = new Float32Array(OBS_FRAME_SIZE);
    let cursor = 0;
    const append = (values, scale = 1) => {
      for (const value of values) frame[cursor++] = value * scale;
    };

    append([0, 0, 0, MASKED_STUDENT_HEIGHT_COMMAND]);
    append(LEFT_PALM_QUATERNION);
    append(RIGHT_PALM_QUATERNION);
    append(LEFT_PALM_POSITION);
    append(RIGHT_PALM_POSITION);
    append([
      qvel[baseDofAddress + 3],
      qvel[baseDofAddress + 4],
      qvel[baseDofAddress + 5],
    ], 0.25);
    append(rotateInverse([
      qpos[baseQposAddress + 3],
      qpos[baseQposAddress + 4],
      qpos[baseQposAddress + 5],
      qpos[baseQposAddress + 6],
    ], [0, 0, -1]));

    for (let policyIndex = 0; policyIndex < N_ACTIONS; policyIndex += 1) {
      const mujocoIndex = policyToMujoco[policyIndex];
      frame[cursor++] = qpos[qposAddresses[mujocoIndex]] - defaultPositionMujoco[mujocoIndex];
    }
    for (let policyIndex = 0; policyIndex < N_ACTIONS; policyIndex += 1) {
      const mujocoIndex = policyToMujoco[policyIndex];
      frame[cursor++] = qvel[dofAddresses[mujocoIndex]] * 0.05;
    }
    append(this.previousAction);
    if (cursor !== OBS_FRAME_SIZE) throw new Error(`Observation frame has ${cursor} values`);
    return frame;
  }

  stackedObservation() {
    const frame = this.observationFrame();
    if (this.observationHistory.length === 0) {
      this.observationHistory = Array.from(
        { length: OBS_HISTORY_LENGTH },
        () => Float32Array.from(frame),
      );
    } else {
      this.observationHistory.push(frame);
      if (this.observationHistory.length > OBS_HISTORY_LENGTH) this.observationHistory.shift();
    }
    const stacked = new Float32Array(OBSERVATION_SIZE);
    this.observationHistory.forEach((historyFrame, index) => {
      for (let i = 0; i < historyFrame.length; i += 1) {
        stacked[index * OBS_FRAME_SIZE + i] = clamp(historyFrame[i], -OBS_CLIP, OBS_CLIP);
      }
    });
    return stacked;
  }

  async inferAction() {
    if (!this.session) throw new Error('ONNX policy is not initialized');
    const observation = this.stackedObservation();
    const feeds = { observation: new ort.Tensor('float32', observation, [1, OBSERVATION_SIZE]) };
    const result = await this.session.run(feeds);
    const rawAction = result.action?.data ?? result[this.session.outputNames[0]]?.data;
    if (!rawAction || rawAction.length !== N_ACTIONS) {
      throw new Error(`Expected ${N_ACTIONS} ONNX actions`);
    }
    for (let i = 0; i < N_ACTIONS; i += 1) {
      const action = clamp(rawAction[i], -ACTION_CLIP, ACTION_CLIP);
      if (!Number.isFinite(action)) throw new Error('Policy produced a non-finite action');
      this.previousAction[i] = action;
    }
    for (let mujocoIndex = 0; mujocoIndex < N_ACTIONS; mujocoIndex += 1) {
      const policyIndex = this.mapping.mujocoToPolicy[mujocoIndex];
      this.targetPositionMujoco[mujocoIndex] = (
        this.mapping.defaultPositionMujoco[mujocoIndex]
        + ACTION_SCALE * this.previousAction[policyIndex]
      );
    }
    this.policyStepCount += 1;
    return this.previousAction;
  }

  applyRobotPd() {
    const { qpos, qvel, ctrl } = this.data;
    const { qposAddresses, dofAddresses, actuatorIds, kp, kd } = this.mapping;
    for (let i = 0; i < N_ACTIONS; i += 1) {
      ctrl[actuatorIds[i]] = (
        kp[i] * (this.targetPositionMujoco[i] - qpos[qposAddresses[i]])
        - kd[i] * qvel[dofAddresses[i]]
      );
    }
  }
}
