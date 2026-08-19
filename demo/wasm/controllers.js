import {
  CONTROLLER_DT,
  DEFAULT_HEIGHT,
  DEFAULT_YAW,
  HEIGHT_LIMITS,
  VELOCITY_X_LIMITS,
  VELOCITY_Y_LIMITS,
  clamp,
  wrapAngle,
} from './constants.js';


function clampNorm2(x, y, limit) {
  const magnitude = Math.hypot(x, y);
  if (magnitude <= limit || magnitude === 0) return [x, y];
  return [x * limit / magnitude, y * limit / magnitude];
}

function quaternionYaw(w, x, y, z) {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}


export class EndpointController {
  constructor(mujoco, model, data, ids, bodyVelocityBuffer) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.ids = ids;
    this.bodyVelocityBuffer = bodyVelocityBuffer;

    this.config = Object.freeze({
      dt: CONTROLLER_DT,
      heightKp: 800,
      heightKd: 35,
      heightForceLimit: 300,
      heightTargetRateLimit: 0.20,
      velocityKp: 30,
      velocityKi: 60,
      velocityKd: 0.1,
      horizontalForceLimit: 100,
      integralForceLimit: 15,
      derivativeCutoffHz: 20,
      velocityTargetSlewLimit: 2,
      velocityErrorDeadband: 0.005,
    });
    this.requestedHeight = DEFAULT_HEIGHT;
    this.requestedVelocityLocal = new Float64Array([0, 0]);
    this.requestedVelocityWorld = new Float64Array([0, 0]);
    this.heightReference = DEFAULT_HEIGHT;
    this.heightReferenceVelocity = 0;
    this.velocityReferenceWorld = new Float64Array([0, 0]);
    this.velocityIntegral = new Float64Array([0, 0]);
    this.filteredAcceleration = new Float64Array([0, 0]);
    this.previousVelocity = new Float64Array([0, 0]);
    this.gravityFeedforward = model.body_mass[ids.carriedBody] * 9.81;
    this.sample = this.zeroSample();
  }

  zeroSample() {
    return {
      position: new Float64Array([0, 0, 0]),
      velocity: new Float64Array([0, 0, 0]),
      velocityReferenceWorld: new Float64Array([0, 0]),
      forceWorld: new Float64Array([0, 0, 0]),
      heightError: 0,
    };
  }

  setTargets({ height, vx, vy } = {}) {
    if (height !== undefined) {
      if (!Number.isFinite(height)) throw new Error('Height target must be finite');
      this.requestedHeight = clamp(height, HEIGHT_LIMITS[0], HEIGHT_LIMITS[1]);
    }
    if (vx !== undefined) {
      if (!Number.isFinite(vx)) throw new Error('Forward velocity target must be finite');
      this.requestedVelocityLocal[0] = clamp(vx, VELOCITY_X_LIMITS[0], VELOCITY_X_LIMITS[1]);
    }
    if (vy !== undefined) {
      if (!Number.isFinite(vy)) throw new Error('Left velocity target must be finite');
      this.requestedVelocityLocal[1] = clamp(vy, VELOCITY_Y_LIMITS[0], VELOCITY_Y_LIMITS[1]);
    }
  }

  headingYaw() {
    const address = this.ids.baseQposAddress;
    const quaternion = [
      this.data.qpos[address + 3],
      this.data.qpos[address + 4],
      this.data.qpos[address + 5],
      this.data.qpos[address + 6],
    ];
    const norm = Math.hypot(...quaternion);
    return quaternionYaw(...quaternion.map((value) => value / norm));
  }

  localToWorld(local) {
    const yaw = this.headingYaw();
    const cosine = Math.cos(yaw);
    const sine = Math.sin(yaw);
    return [cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1]];
  }

  worldToLocal(world) {
    const yaw = this.headingYaw();
    const cosine = Math.cos(yaw);
    const sine = Math.sin(yaw);
    return [cosine * world[0] + sine * world[1], -sine * world[0] + cosine * world[1]];
  }

  endpointState() {
    this.mujoco.mj_objectVelocity(
      this.model,
      this.data,
      this.mujoco.mjtObj.mjOBJ_BODY.value,
      this.ids.carriedBody,
      this.bodyVelocityBuffer,
      0,
    );
    const spatial = this.bodyVelocityBuffer.GetView();
    const positionAddress = 3 * this.ids.controllerSite;
    return {
      position: new Float64Array([
        this.data.site_xpos[positionAddress],
        this.data.site_xpos[positionAddress + 1],
        this.data.site_xpos[positionAddress + 2],
      ]),
      velocity: new Float64Array([spatial[3], spatial[4], spatial[5]]),
    };
  }

  reset() {
    const state = this.endpointState();
    this.heightReference = state.position[2];
    this.heightReferenceVelocity = 0;
    const requestedWorld = this.localToWorld(this.requestedVelocityLocal);
    this.requestedVelocityWorld.set(requestedWorld);
    this.velocityReferenceWorld.set(state.velocity.subarray(0, 2));
    this.velocityIntegral.fill(0);
    this.filteredAcceleration.fill(0);
    this.previousVelocity.set(state.velocity.subarray(0, 2));
    this.sample = this.zeroSample();
    this.sample.position.set(state.position);
    this.sample.velocity.set(state.velocity);
  }

  compute() {
    const cfg = this.config;
    const requestedWorld = this.localToWorld(this.requestedVelocityLocal);
    this.requestedVelocityWorld.set(requestedWorld);

    const heightDelta = clamp(
      this.requestedHeight - this.heightReference,
      -cfg.heightTargetRateLimit * cfg.dt,
      cfg.heightTargetRateLimit * cfg.dt,
    );
    this.heightReference += heightDelta;
    this.heightReferenceVelocity = heightDelta / cfg.dt;
    for (let axis = 0; axis < 2; axis += 1) {
      const delta = clamp(
        this.requestedVelocityWorld[axis] - this.velocityReferenceWorld[axis],
        -cfg.velocityTargetSlewLimit * cfg.dt,
        cfg.velocityTargetSlewLimit * cfg.dt,
      );
      this.velocityReferenceWorld[axis] += delta;
    }

    const state = this.endpointState();
    const heightError = this.heightReference - state.position[2];
    const heightForce = clamp(
      cfg.heightKp * heightError
      + cfg.heightKd * (this.heightReferenceVelocity - state.velocity[2])
      + this.gravityFeedforward,
      -cfg.heightForceLimit,
      cfg.heightForceLimit,
    );

    const alpha = Math.exp(-2 * Math.PI * cfg.derivativeCutoffHz * cfg.dt);
    const velocityError = new Float64Array(2);
    const integrationError = new Float64Array(2);
    const proportional = new Float64Array(2);
    const derivative = new Float64Array(2);
    for (let axis = 0; axis < 2; axis += 1) {
      const acceleration = (state.velocity[axis] - this.previousVelocity[axis]) / cfg.dt;
      this.filteredAcceleration[axis] = (
        alpha * this.filteredAcceleration[axis]
        + (1 - alpha) * acceleration
      );
      this.previousVelocity[axis] = state.velocity[axis];
      velocityError[axis] = this.velocityReferenceWorld[axis] - state.velocity[axis];
      integrationError[axis] = (
        Math.abs(velocityError[axis]) < cfg.velocityErrorDeadband ? 0 : velocityError[axis]
      );
      proportional[axis] = cfg.velocityKp * velocityError[axis];
      derivative[axis] = -cfg.velocityKd * this.filteredAcceleration[axis];
    }

    const currentIntegralForce = [
      cfg.velocityKi * this.velocityIntegral[0],
      cfg.velocityKi * this.velocityIntegral[1],
    ];
    const unsaturatedForce = [
      proportional[0] + currentIntegralForce[0] + derivative[0],
      proportional[1] + currentIntegralForce[1] + derivative[1],
    ];
    const unsaturatedMagnitude = Math.hypot(...unsaturatedForce);
    if (unsaturatedMagnitude >= cfg.horizontalForceLimit && unsaturatedMagnitude > 0) {
      const direction = unsaturatedForce.map((value) => value / unsaturatedMagnitude);
      const outwardError = integrationError[0] * direction[0] + integrationError[1] * direction[1];
      if (outwardError > 0) {
        integrationError[0] -= outwardError * direction[0];
        integrationError[1] -= outwardError * direction[1];
      }
    }
    for (let axis = 0; axis < 2; axis += 1) {
      this.velocityIntegral[axis] += integrationError[axis] * cfg.dt;
    }
    const integralForce = clampNorm2(
      cfg.velocityKi * this.velocityIntegral[0],
      cfg.velocityKi * this.velocityIntegral[1],
      cfg.integralForceLimit,
    );
    this.velocityIntegral[0] = integralForce[0] / cfg.velocityKi;
    this.velocityIntegral[1] = integralForce[1] / cfg.velocityKi;
    const horizontal = clampNorm2(
      proportional[0] + integralForce[0] + derivative[0],
      proportional[1] + integralForce[1] + derivative[1],
      cfg.horizontalForceLimit,
    );

    this.sample = {
      position: state.position,
      velocity: state.velocity,
      velocityReferenceWorld: Float64Array.from(this.velocityReferenceWorld),
      forceWorld: new Float64Array([horizontal[0], horizontal[1], heightForce]),
      heightError,
    };
    return this.sample;
  }

  apply(sample = this.sample) {
    this.mujoco.mj_applyFT(
      this.model,
      this.data,
      Array.from(sample.forceWorld),
      [0, 0, 0],
      Array.from(sample.position),
      this.ids.carriedBody,
      this.data.qfrc_applied,
    );
  }
}


export class YawController {
  constructor(mujoco, model, data, ids, bodyVelocityBuffer) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.ids = ids;
    this.bodyVelocityBuffer = bodyVelocityBuffer;
    this.config = Object.freeze({
      dt: CONTROLLER_DT,
      kp: 40,
      kd: 4,
      torqueLimit: 10,
      targetRateLimit: 45 * Math.PI / 180,
    });
    this.requestedYaw = DEFAULT_YAW;
    this.referenceYaw = DEFAULT_YAW;
    this.referenceRate = 0;
    this.sample = this.zeroSample();
  }

  zeroSample() {
    return {
      measuredYaw: DEFAULT_YAW,
      yawError: 0,
      torqueScalar: 0,
      torqueWorld: new Float64Array([0, 0, 0]),
      currentVector: new Float64Array([0, -1, 0]),
      targetVector: new Float64Array([0, -1, 0]),
    };
  }

  setTarget(yaw) {
    if (!Number.isFinite(yaw)) throw new Error('Yaw target must be finite');
    this.requestedYaw = wrapAngle(yaw);
  }

  sitePosition(siteId) {
    const address = 3 * siteId;
    return [
      this.data.site_xpos[address],
      this.data.site_xpos[address + 1],
      this.data.site_xpos[address + 2],
    ];
  }

  barVector() {
    const positive = this.sitePosition(this.ids.positiveEndpointSite);
    const negative = this.sitePosition(this.ids.negativeEndpointSite);
    return new Float64Array([
      negative[0] - positive[0],
      negative[1] - positive[1],
      negative[2] - positive[2],
    ]);
  }

  measuredYaw() {
    const vector = this.barVector();
    if (Math.hypot(vector[0], vector[1]) < 1e-6) throw new Error('Bar yaw is undefined');
    return Math.atan2(vector[1], vector[0]);
  }

  reset() {
    this.referenceYaw = this.measuredYaw();
    this.referenceRate = 0;
    this.sample = this.zeroSample();
    this.sample.measuredYaw = this.referenceYaw;
    this.sample.currentVector = this.barVector();
  }

  compute() {
    const maximumDelta = this.config.targetRateLimit * this.config.dt;
    const appliedDelta = clamp(
      wrapAngle(this.requestedYaw - this.referenceYaw),
      -maximumDelta,
      maximumDelta,
    );
    this.referenceYaw = wrapAngle(this.referenceYaw + appliedDelta);
    this.referenceRate = appliedDelta / this.config.dt;

    const vector = this.barVector();
    const measuredYaw = Math.atan2(vector[1], vector[0]);
    this.mujoco.mj_objectVelocity(
      this.model,
      this.data,
      this.mujoco.mjtObj.mjOBJ_BODY.value,
      this.ids.carriedBody,
      this.bodyVelocityBuffer,
      0,
    );
    const spatial = this.bodyVelocityBuffer.GetView();
    const vectorRate = [
      spatial[1] * vector[2] - spatial[2] * vector[1],
      spatial[2] * vector[0] - spatial[0] * vector[2],
      spatial[0] * vector[1] - spatial[1] * vector[0],
    ];
    const horizontalNormSquared = vector[0] ** 2 + vector[1] ** 2;
    const yawRate = (
      vector[0] * vectorRate[1] - vector[1] * vectorRate[0]
    ) / horizontalNormSquared;
    const yawError = wrapAngle(this.referenceYaw - measuredYaw);
    const torqueScalar = clamp(
      this.config.kp * yawError + this.config.kd * (this.referenceRate - yawRate),
      -this.config.torqueLimit,
      this.config.torqueLimit,
    );
    const vectorNorm = Math.hypot(...vector);
    const unit = Array.from(vector, (value) => value / vectorNorm);
    const projection = unit[2];
    const yawAxis = [-projection * unit[0], -projection * unit[1], 1 - projection * unit[2]];
    const torqueWorld = new Float64Array(yawAxis.map((value) => torqueScalar * value));
    this.sample = {
      measuredYaw,
      yawError,
      torqueScalar,
      torqueWorld,
      currentVector: vector,
      targetVector: new Float64Array([Math.cos(this.referenceYaw), Math.sin(this.referenceYaw), 0]),
    };
    return this.sample;
  }

  apply(sample = this.sample) {
    const address = 3 * this.ids.controllerSite;
    const position = [
      this.data.site_xpos[address],
      this.data.site_xpos[address + 1],
      this.data.site_xpos[address + 2],
    ];
    this.mujoco.mj_applyFT(
      this.model,
      this.data,
      [0, 0, 0],
      Array.from(sample.torqueWorld),
      position,
      this.ids.carriedBody,
      this.data.qfrc_applied,
    );
  }
}
