import loadMujoco from './vendor/mujoco/mujoco.js';

import {
  BAR_INITIAL_CENTER,
  CONTROLLER_HZ,
  DEFAULT_HEIGHT,
  DEFAULT_POSITION_BY_NAME,
  DEFAULT_YAW,
  HEIGHT_LIMITS,
  N_ACTIONS,
  PHYSICS_HZ,
  POLICY_DECIMATION,
  POLICY_HZ,
  POLICY_JOINT_ORDER,
  ROBOT_INITIAL_POSITION,
  VELOCITY_X_LIMITS,
  VELOCITY_Y_LIMITS,
  clamp,
  gainsForJoint,
  wrapAngle,
} from './constants.js';
import { EndpointController, YawController } from './controllers.js';
import { StudentPolicy, createPolicySession } from './policy.js';
import { MujocoRenderer } from './renderer.js';


const SCENE_PATH = 'centered_fixed_bar_scene.xml';
const ROBOT_XML_PATH = 'g1/g1_29dof_centered_fixed_bar.xml';
const MODEL_ROOT = '/cola/model';
const MAX_CATCHUP_STEPS = 24;


function degrees(radians) {
  return radians * 180 / Math.PI;
}

function createDirectoryTree(fs) {
  fs.mkdirTree(`${MODEL_ROOT}/g1/meshes`);
}

async function fetchRequired(responsePromise, label) {
  const response = await responsePromise;
  if (!response.ok) throw new Error(`Failed to load ${label} (${response.status})`);
  return response;
}


export class ColaBrowserRuntime extends EventTarget {
  constructor(canvas, progressCallback = () => {}) {
    super();
    this.canvas = canvas;
    this.progressCallback = progressCallback;
    this.ready = false;
    this.runtimeError = null;
    this.loadingMessage = 'Loading MuJoCo WebAssembly';
    this.paused = false;
    this.running = false;
    this.physicsStep = 0;
    this.controllerPhase = 0;
    this.accumulator = 0;
    this.lastFrameTimestamp = null;
    this.controllerOverlays = { velocity: false, height: false, torque: false };
    this.controllersEnabled = true;
    this.animationFrame = null;
  }

  report(message, progress = null) {
    this.loadingMessage = message;
    this.progressCallback({ message, progress });
  }

  async loadModelAssets() {
    const assetRoot = new URL('./assets/model/', import.meta.url);
    const sceneResponse = await fetchRequired(fetch(new URL(SCENE_PATH, assetRoot)), SCENE_PATH);
    const robotResponse = await fetchRequired(fetch(new URL(ROBOT_XML_PATH, assetRoot)), ROBOT_XML_PATH);
    const [sceneXml, robotXml] = await Promise.all([sceneResponse.text(), robotResponse.text()]);
    const meshPaths = Array.from(
      robotXml.matchAll(/\bfile="([^"]+\.STL)"/gi),
      (match) => `g1/${match[1]}`,
    );
    const files = [
      { path: SCENE_PATH, bytes: new TextEncoder().encode(sceneXml) },
      { path: ROBOT_XML_PATH, bytes: new TextEncoder().encode(robotXml) },
    ];
    let completed = 0;
    const meshes = await Promise.all(meshPaths.map(async (path) => {
      const response = await fetchRequired(fetch(new URL(path, assetRoot)), path);
      const bytes = new Uint8Array(await response.arrayBuffer());
      completed += 1;
      this.report(`Loading robot meshes (${completed}/${meshPaths.length})`, completed / meshPaths.length);
      return { path, bytes };
    }));
    return [...files, ...meshes];
  }

  async initialize({ start = true } = {}) {
    try {
      this.report('Loading the student policy', 0);
      const policySession = await createPolicySession(
        new URL(
          './assets/policy-loop-static-three-jitter-phase3-model-5000-2433dac7.onnx',
          import.meta.url,
        ).href,
      );
      this.report('Loading MuJoCo WebAssembly', 0.18);
      const [mujoco, modelFiles] = await Promise.all([
        loadMujoco({
          locateFile: (path) => new URL(`./vendor/mujoco/${path}`, import.meta.url).href,
        }),
        this.loadModelAssets(),
      ]);
      this.mujoco = mujoco;
      createDirectoryTree(mujoco.FS);
      for (const file of modelFiles) {
        mujoco.FS.writeFile(`${MODEL_ROOT}/${file.path}`, file.bytes);
      }

      this.report('Compiling the MuJoCo scene', 0.82);
      this.model = mujoco.MjModel.mj_loadXML(`${MODEL_ROOT}/${SCENE_PATH}`);
      if (!this.model) throw new Error('MuJoCo failed to compile the scene');
      this.data = new mujoco.MjData(this.model);
      if (!this.data) throw new Error('MuJoCo failed to allocate simulation data');
      if (Math.abs(this.model.opt.timestep - 1 / PHYSICS_HZ) > 1e-12) {
        throw new Error(`Scene timestep ${this.model.opt.timestep} does not match ${PHYSICS_HZ} Hz`);
      }

      this.mapping = this.buildRobotMapping();
      this.ids = this.buildIds();
      this.bodyVelocityBufferEndpoint = new mujoco.DoubleBuffer(6);
      this.bodyVelocityBufferYaw = new mujoco.DoubleBuffer(6);
      this.policy = new StudentPolicy(this.model, this.data, this.mapping);
      this.policy.initialize(policySession);

      this.endpointController = new EndpointController(
        mujoco,
        this.model,
        this.data,
        this.ids,
        this.bodyVelocityBufferEndpoint,
      );
      this.yawController = new YawController(
        mujoco,
        this.model,
        this.data,
        this.ids,
        this.bodyVelocityBufferYaw,
      );
      this.endpointController.setTargets({ height: DEFAULT_HEIGHT, vx: 0, vy: 0 });
      this.yawController.setTarget(DEFAULT_YAW);
      this.resetSimulation();

      this.report('Building the browser renderer', 0.96);
      this.renderer = new MujocoRenderer(this.canvas, mujoco, this.model, this.data, this.ids);
      this.ready = true;
      this.report('Ready', 1);
      if (start) this.start();
      return this;
    } catch (error) {
      this.runtimeError = error instanceof Error ? error.message : String(error);
      console.error('COLA browser runtime initialization failed', error);
      throw error;
    }
  }

  objectId(kind, name) {
    const id = this.mujoco.mj_name2id(this.model, kind.value, name);
    if (id < 0) throw new Error(`MuJoCo object not found: ${name}`);
    return id;
  }

  buildRobotMapping() {
    const hingeType = this.mujoco.mjtJoint.mjJNT_HINGE.value;
    const objectJoint = this.mujoco.mjtObj.mjOBJ_JOINT;
    const jointIds = [];
    const jointNames = [];
    for (let jointId = 0; jointId < this.model.njnt; jointId += 1) {
      if (this.model.jnt_type[jointId] !== hingeType) continue;
      jointIds.push(jointId);
      jointNames.push(this.mujoco.mj_id2name(this.model, objectJoint.value, jointId));
    }
    if (jointNames.length !== N_ACTIONS) {
      throw new Error(`Expected ${N_ACTIONS} hinge joints, found ${jointNames.length}`);
    }
    const expected = new Set(POLICY_JOINT_ORDER);
    const actual = new Set(jointNames);
    if (expected.size !== actual.size || [...expected].some((name) => !actual.has(name))) {
      throw new Error('Robot joints do not match the policy joint set');
    }
    const qposAddresses = Int32Array.from(jointIds, (id) => this.model.jnt_qposadr[id]);
    const dofAddresses = Int32Array.from(jointIds, (id) => this.model.jnt_dofadr[id]);
    const nameToMujoco = new Map(jointNames.map((name, index) => [name, index]));
    const policyToMujoco = Int32Array.from(POLICY_JOINT_ORDER, (name) => nameToMujoco.get(name));
    const mujocoToPolicy = new Int32Array(N_ACTIONS);
    policyToMujoco.forEach((mujocoIndex, policyIndex) => { mujocoToPolicy[mujocoIndex] = policyIndex; });
    const defaultPositionMujoco = Float64Array.from(
      jointNames,
      (name) => DEFAULT_POSITION_BY_NAME[name] ?? 0,
    );
    const kp = new Float64Array(N_ACTIONS);
    const kd = new Float64Array(N_ACTIONS);
    jointNames.forEach((name, index) => {
      [kp[index], kd[index]] = gainsForJoint(name);
    });
    const jointIdToActuator = new Map();
    for (let actuatorId = 0; actuatorId < this.model.nu; actuatorId += 1) {
      jointIdToActuator.set(this.model.actuator_trnid[2 * actuatorId], actuatorId);
    }
    const actuatorIds = Int32Array.from(jointIds, (jointId) => {
      const actuatorId = jointIdToActuator.get(jointId);
      if (actuatorId === undefined) throw new Error(`No actuator for joint ${jointId}`);
      return actuatorId;
    });
    const baseJoint = this.objectId(objectJoint, 'floating_base_joint');
    const barJoint = this.objectId(objectJoint, 'object_freejoint');
    return {
      jointIds,
      jointNames,
      qposAddresses,
      dofAddresses,
      policyToMujoco,
      mujocoToPolicy,
      defaultPositionMujoco,
      kp,
      kd,
      actuatorIds,
      baseQposAddress: this.model.jnt_qposadr[baseJoint],
      baseDofAddress: this.model.jnt_dofadr[baseJoint],
      barQposAddress: this.model.jnt_qposadr[barJoint],
      barDofAddress: this.model.jnt_dofadr[barJoint],
    };
  }

  buildIds() {
    return {
      baseQposAddress: this.mapping.baseQposAddress,
      carriedBody: this.objectId(this.mujoco.mjtObj.mjOBJ_BODY, 'carried_object'),
      controllerSite: this.objectId(this.mujoco.mjtObj.mjOBJ_SITE, 'controller_point'),
      negativeEndpointSite: this.objectId(this.mujoco.mjtObj.mjOBJ_SITE, 'negative_y_endpoint'),
      positiveEndpointSite: this.objectId(this.mujoco.mjtObj.mjOBJ_SITE, 'positive_y_endpoint'),
    };
  }

  resetSimulation() {
    if (!this.model || !this.data) return;
    this.mujoco.mj_resetData(this.model, this.data);
    const { baseQposAddress, barQposAddress, qposAddresses, defaultPositionMujoco } = this.mapping;
    this.data.qpos.set([...ROBOT_INITIAL_POSITION, 1, 0, 0, 0], baseQposAddress);
    qposAddresses.forEach((address, index) => { this.data.qpos[address] = defaultPositionMujoco[index]; });
    this.data.qpos.set([...BAR_INITIAL_CENTER, 1, 0, 0, 0], barQposAddress);
    this.data.qvel.fill(0);
    this.data.ctrl.fill(0);
    this.data.qfrc_applied.fill(0);
    this.mujoco.mj_forward(this.model, this.data);
    this.policy?.reset();
    this.endpointController?.reset();
    this.yawController?.reset();
    this.physicsStep = 0;
    this.controllerPhase = 0;
    this.accumulator = 0;
    this.runtimeError = null;
  }

  async runPhysicsStep() {
    if (this.physicsStep % POLICY_DECIMATION === 0) await this.policy.inferAction();
    this.data.qfrc_applied.fill(0);
    this.policy.applyRobotPd();
    let controllerDue = this.physicsStep === 0;
    if (!controllerDue) {
      this.controllerPhase += CONTROLLER_HZ;
      if (this.controllerPhase >= PHYSICS_HZ) {
        this.controllerPhase -= PHYSICS_HZ;
        controllerDue = true;
      }
    }
    if (controllerDue) {
      this.endpointController.compute();
      this.yawController.compute();
    }
    if (this.controllersEnabled) {
      this.endpointController.apply();
      this.yawController.apply();
    }
    this.mujoco.mj_step(this.model, this.data);
    this.physicsStep += 1;
  }

  start() {
    if (this.running) return;
    this.running = true;
    const frame = async (timestamp) => {
      if (!this.running) return;
      try {
        if (this.lastFrameTimestamp === null) this.lastFrameTimestamp = timestamp;
        const elapsed = Math.min(0.05, Math.max(0, (timestamp - this.lastFrameTimestamp) / 1000));
        this.lastFrameTimestamp = timestamp;
        if (!this.paused && !this.runtimeError) this.accumulator += elapsed;
        let steps = 0;
        while (this.accumulator >= 1 / PHYSICS_HZ && steps < MAX_CATCHUP_STEPS) {
          await this.runPhysicsStep();
          this.accumulator -= 1 / PHYSICS_HZ;
          steps += 1;
        }
        if (steps === MAX_CATCHUP_STEPS) this.accumulator = Math.min(this.accumulator, 1 / PHYSICS_HZ);
        this.renderer.render();
      } catch (error) {
        this.runtimeError = error instanceof Error ? error.message : String(error);
        console.error('COLA browser simulation failed', error);
      }
      this.animationFrame = requestAnimationFrame(frame);
    };
    this.animationFrame = requestAnimationFrame(frame);
  }

  stop() {
    this.running = false;
    if (this.animationFrame !== null) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
  }

  command(command) {
    const normalized = command.toLowerCase();
    const velocityStep = 0.05;
    if (normalized === 'w') this.endpointController.setTargets({ vx: this.endpointController.requestedVelocityLocal[0] + velocityStep });
    else if (normalized === 's') this.endpointController.setTargets({ vx: this.endpointController.requestedVelocityLocal[0] - velocityStep });
    else if (normalized === 'a') this.endpointController.setTargets({ vy: this.endpointController.requestedVelocityLocal[1] + velocityStep });
    else if (normalized === 'd') this.endpointController.setTargets({ vy: this.endpointController.requestedVelocityLocal[1] - velocityStep });
    else if (normalized === 'i') this.endpointController.setTargets({ height: this.endpointController.requestedHeight + 0.01 });
    else if (normalized === 'k') this.endpointController.setTargets({ height: this.endpointController.requestedHeight - 0.01 });
    else if (normalized === 'j') this.yawController.setTarget(this.yawController.requestedYaw - 5 * Math.PI / 180);
    else if (normalized === 'l') this.yawController.setTarget(this.yawController.requestedYaw + 5 * Math.PI / 180);
    else if (normalized === 'c') this.yawController.setTarget(DEFAULT_YAW);
    else if (normalized === 'x') this.endpointController.setTargets({ vx: 0, vy: 0 });
    else if (normalized === 'r') this.resetSimulation();
    else if (normalized === 'toggle-pause') this.paused = !this.paused;
    else throw new Error(`Unknown command: ${command}`);
    return this.getState();
  }

  setVelocity(vx, vy) {
    this.endpointController.setTargets({ vx, vy });
    return this.getState();
  }

  setHeight(height) {
    this.endpointController.setTargets({ height });
    return this.getState();
  }

  setYawDegrees(yawDegrees) {
    this.yawController.setTarget(yawDegrees * Math.PI / 180);
    return this.getState();
  }

  resetTargets() {
    this.endpointController.setTargets({ height: DEFAULT_HEIGHT, vx: 0, vy: 0 });
    this.yawController.setTarget(DEFAULT_YAW);
    return this.getState();
  }

  toggleCameraFollow() {
    this.renderer.follow = !this.renderer.follow;
    return this.getState();
  }

  toggleControllerOverlay(controller) {
    if (!(controller in this.controllerOverlays)) throw new Error(`Unknown controller overlay: ${controller}`);
    this.controllerOverlays[controller] = !this.controllerOverlays[controller];
    return this.getState();
  }

  getState() {
    if (!this.ready) {
      return {
        ready: false,
        runtime_error: this.runtimeError,
        loading_message: this.loadingMessage,
      };
    }
    const endpointState = this.endpointController.endpointState();
    const localVelocity = this.endpointController.worldToLocal(endpointState.velocity);
    const localForce = this.endpointController.worldToLocal(this.endpointController.sample.forceWorld);
    const baseHeight = this.data.qpos[this.mapping.baseQposAddress + 2];
    const actualYaw = this.yawController.measuredYaw();
    const camera = this.renderer.cameraState();
    const healthy = (
      Number.isFinite(baseHeight)
      && baseHeight > 0.35
      && Array.from(this.data.qpos).every(Number.isFinite)
    );
    return {
      ready: true,
      healthy,
      runtime_error: this.runtimeError,
      time: this.data.time,
      paused: this.paused,
      controllers_enabled: this.controllersEnabled,
      vectors_visible: Object.values(this.controllerOverlays).some(Boolean),
      overlays: { ...this.controllerOverlays },
      base: { height: baseHeight },
      actual: {
        height: endpointState.position[2],
        vx: localVelocity[0],
        vy: localVelocity[1],
        yaw_deg: degrees(actualYaw),
      },
      target: {
        height: this.endpointController.requestedHeight,
        vx: this.endpointController.requestedVelocityLocal[0],
        vy: this.endpointController.requestedVelocityLocal[1],
        yaw_deg: degrees(this.yawController.requestedYaw),
      },
      reference: {
        height: this.endpointController.heightReference,
        yaw_deg: degrees(this.yawController.referenceYaw),
      },
      wrench_local: {
        fx: localForce[0],
        fy: localForce[1],
        fz: this.endpointController.sample.forceWorld[2],
        tz: this.yawController.sample.torqueScalar,
      },
      limits: {
        height: [...HEIGHT_LIMITS],
        vx: [...VELOCITY_X_LIMITS],
        vy: [...VELOCITY_Y_LIMITS],
      },
      rates: {
        physics_hz: PHYSICS_HZ,
        controller_hz: CONTROLLER_HZ,
        policy_hz: POLICY_HZ,
      },
      controller: {
        height_kp: this.endpointController.config.heightKp,
        height_kd: this.endpointController.config.heightKd,
        height_force_limit: this.endpointController.config.heightForceLimit,
        velocity_kp: this.endpointController.config.velocityKp,
        velocity_ki: this.endpointController.config.velocityKi,
        velocity_kd: this.endpointController.config.velocityKd,
        horizontal_force_limit: this.endpointController.config.horizontalForceLimit,
        yaw_kp: this.yawController.config.kp,
        yaw_kd: this.yawController.config.kd,
        yaw_torque_limit: this.yawController.config.torqueLimit,
      },
      camera_follow: this.renderer.follow,
      camera,
      checkpoint: 'policy-loop-static-three-jitter-phase3-model-5000-2433dac7.onnx',
    };
  }

  dispose() {
    this.stop();
    this.renderer?.dispose();
    this.bodyVelocityBufferEndpoint?.delete();
    this.bodyVelocityBufferYaw?.delete();
    this.data?.delete();
    this.model?.delete();
  }
}
