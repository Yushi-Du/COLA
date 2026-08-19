export const PHYSICS_HZ = 1000;
export const CONTROLLER_HZ = 400;
export const POLICY_HZ = 50;
export const POLICY_DECIMATION = PHYSICS_HZ / POLICY_HZ;
export const PHYSICS_DT = 1 / PHYSICS_HZ;
export const CONTROLLER_DT = 1 / CONTROLLER_HZ;

export const N_ACTIONS = 29;
export const OBS_FRAME_SIZE = 111;
export const OBS_HISTORY_LENGTH = 25;
export const OBSERVATION_SIZE = OBS_FRAME_SIZE * OBS_HISTORY_LENGTH;
export const ACTION_SCALE = 0.25;
export const OBS_CLIP = 100;
export const ACTION_CLIP = 100;

export const ROBOT_INITIAL_POSITION = Object.freeze([0, 0, 0.8]);
export const BAR_INITIAL_CENTER = Object.freeze([0.298, 0, 0.924]);
export const DEFAULT_HEIGHT = 0.70;
export const DEFAULT_YAW = -0.5 * Math.PI;
export const MASKED_STUDENT_HEIGHT_COMMAND = 0.78;

export const HEIGHT_LIMITS = Object.freeze([0.55, 0.85]);
export const VELOCITY_X_LIMITS = Object.freeze([-0.5, 0.5]);
export const VELOCITY_Y_LIMITS = Object.freeze([-0.5, 0.5]);

export const POLICY_JOINT_ORDER = Object.freeze([
  'left_hip_pitch_joint',
  'right_hip_pitch_joint',
  'waist_yaw_joint',
  'left_hip_roll_joint',
  'right_hip_roll_joint',
  'waist_roll_joint',
  'left_hip_yaw_joint',
  'right_hip_yaw_joint',
  'waist_pitch_joint',
  'left_knee_joint',
  'right_knee_joint',
  'left_shoulder_pitch_joint',
  'right_shoulder_pitch_joint',
  'left_ankle_pitch_joint',
  'right_ankle_pitch_joint',
  'left_shoulder_roll_joint',
  'right_shoulder_roll_joint',
  'left_ankle_roll_joint',
  'right_ankle_roll_joint',
  'left_shoulder_yaw_joint',
  'right_shoulder_yaw_joint',
  'left_elbow_joint',
  'right_elbow_joint',
  'left_wrist_roll_joint',
  'right_wrist_roll_joint',
  'left_wrist_pitch_joint',
  'right_wrist_pitch_joint',
  'left_wrist_yaw_joint',
  'right_wrist_yaw_joint',
]);

export const DEFAULT_POSITION_BY_NAME = Object.freeze({
  left_hip_pitch_joint: -0.20,
  right_hip_pitch_joint: -0.20,
  left_knee_joint: 0.42,
  right_knee_joint: 0.42,
  left_ankle_pitch_joint: -0.23,
  right_ankle_pitch_joint: -0.23,
  left_wrist_roll_joint: -0.5 * Math.PI,
  right_wrist_roll_joint: 0.5 * Math.PI,
});

export const LEFT_PALM_QUATERNION = Object.freeze([0.707, -0.707, 0, 0]);
export const RIGHT_PALM_QUATERNION = Object.freeze([0.707, 0.707, 0, 0]);
export const LEFT_PALM_POSITION = Object.freeze([0.2413, 0.1517, 0.0952]);
export const RIGHT_PALM_POSITION = Object.freeze([0.2413, -0.1516, 0.0952]);

export function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

export function wrapAngle(angle) {
  return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

export function gainsForJoint(name) {
  if (name.includes('hip_roll')) return [99.09842777666113, 6.3088018534966395];
  if (name.includes('hip_yaw') || name.includes('hip_pitch')) return [40.17923847137318, 2.5578897650279457];
  if (name.includes('knee')) return [99.09842777666113, 6.3088018534966395];
  if (name.includes('waist_yaw')) return [40.17923847137318, 2.5578897650279457];
  if (name.includes('waist_roll') || name.includes('waist_pitch')) return [28.50124619574858, 1.814445686584846];
  if (name.includes('ankle')) return [28.50124619574858, 1.814445686584846];
  if (name.includes('shoulder') || name.includes('elbow')) return [14.25062309787429, 0.907222843292423];
  if (name.includes('wrist_roll')) return [14.25062309787429, 0.907222843292423];
  if (name.includes('wrist_pitch') || name.includes('wrist_yaw')) return [16.77832748089279, 1.06814150219];
  throw new Error(`No SMOKV3SP gains are defined for ${name}`);
}
