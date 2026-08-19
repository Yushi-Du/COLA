import { ColaBrowserRuntime } from './runtime.js?v=model5000-2433dac7';

const elements = {
  connection: document.querySelector('#connection-pill'),
  loading: document.querySelector('#loading-panel'),
  stream: document.querySelector('#simulation-stream'),
  viewport: document.querySelector('#viewport'),
  pause: document.querySelector('#pause-button'),
  cameraFollow: document.querySelector('#camera-follow'),
  dock: document.querySelector('.control-dock'),
  controllerVisuals: document.querySelector('.controller-visuals'),
  velocityTrack: document.querySelector('#velocity-track'),
  velocityThumb: document.querySelector('#velocity-thumb'),
  height: document.querySelector('#height-slider'),
  poseCanvas: document.querySelector('#target-vector-canvas'),
  lateralForceCanvas: document.querySelector('#lateral-force-canvas'),
  heightForceCanvas: document.querySelector('#height-force-canvas'),
  torqueHistoryCanvas: document.querySelector('#torque-history-canvas'),
  toast: document.querySelector('#toast'),
};
const SMOKE_MODE = new URLSearchParams(window.location.search).has('smoke');

let latestState = null;
let toastTimer = null;
let velocityRequestPending = false;
let queuedVelocityTarget = null;
let velocityFlushTimer = null;
let heightRequestPending = false;
let queuedHeightTarget = null;
let heightFlushTimer = null;
let yawRequestPending = false;
let queuedYawTarget = null;
let yawFlushTimer = null;
let velocityPointerId = null;
let posePointerId = null;
let previewYawDeg = null;
const activeSliders = new Set();
const CONTROLLER_HISTORY_SECONDS = 3;
const CONTROLLER_SAMPLE_INTERVAL_SECONDS = 0.04;
let controllerHistory = [];
let browserRuntime = null;
let runtimePromise = null;

const number = (value, digits = 2, sign = false) => {
  if (!Number.isFinite(value)) return '—';
  const result = value.toFixed(digits);
  return sign && value >= 0 ? `+${result}` : result;
};

const normalizeDegrees = (value) => ((value + 180) % 360 + 360) % 360 - 180;

const degrees = (value) => {
  if (!Number.isFinite(value)) return '—';
  const normalized = normalizeDegrees(value);
  return `${normalized < 0 ? '−' : '+'}${Math.abs(normalized).toFixed(0)}`;
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('visible');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('visible'), 1600);
}

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
}

function velocityLimits() {
  return {
    vx: latestState?.limits?.vx || [-0.5, 0.5],
    vy: latestState?.limits?.vy || [-0.5, 0.5],
  };
}

function updateVelocityThumb(vx, vy) {
  const limits = velocityLimits();
  const x = 1 - (vy - limits.vy[0]) / (limits.vy[1] - limits.vy[0]);
  const y = 1 - (vx - limits.vx[0]) / (limits.vx[1] - limits.vx[0]);
  elements.velocityThumb.style.left = `${Math.max(0, Math.min(1, x)) * 100}%`;
  elements.velocityThumb.style.top = `${Math.max(0, Math.min(1, y)) * 100}%`;
}

function poseFrame(width, height) {
  const unit = Math.min(width, height) * 0.53;
  const azimuth = (latestState?.camera?.azimuth ?? 138) * Math.PI / 180;
  const elevation = (latestState?.camera?.elevation ?? -13) * Math.PI / 180;
  // Match the actual Three.js camera basis used by the MuJoCo viewport.
  // cameraState() represents the horizontal camera offset as
  // (sin(azimuth), -cos(azimuth)).
  const right = {
    x: Math.cos(azimuth),
    y: Math.sin(azimuth),
    z: 0,
  };
  const up = {
    x: Math.sin(azimuth) * Math.sin(elevation),
    y: -Math.cos(azimuth) * Math.sin(elevation),
    z: Math.cos(elevation),
  };
  return {
    width,
    height,
    origin: { x: width * 0.50, y: height * 0.62 },
    ex: { x: unit * right.x, y: -unit * up.x },
    ey: { x: unit * right.y, y: -unit * up.y },
    ez: { x: unit * right.z, y: -unit * up.z },
  };
}

function projectPoint(frame, point) {
  return {
    x: frame.origin.x + point.x * frame.ex.x + point.y * frame.ey.x + point.z * frame.ez.x,
    y: frame.origin.y + point.x * frame.ex.y + point.y * frame.ey.y + point.z * frame.ez.y,
  };
}

function line(ctx, from, to, color, width = 1, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.restore();
}

function drawAxis(ctx, frame, endpoint, color, label) {
  const start = projectPoint(frame, { x: 0, y: 0, z: 0 });
  const end = projectPoint(frame, endpoint);
  line(ctx, start, end, color, 1.6);
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const head = 5;
  line(ctx, end, {
    x: end.x - head * Math.cos(angle - 0.55),
    y: end.y - head * Math.sin(angle - 0.55),
  }, color, 1.6);
  line(ctx, end, {
    x: end.x - head * Math.cos(angle + 0.55),
    y: end.y - head * Math.sin(angle + 0.55),
  }, color, 1.6);
  ctx.fillStyle = color;
  ctx.font = '600 11px DM Mono, monospace';
  ctx.fillText(label, end.x + 4, end.y - 3);
}

function cuboidVertices(yawDeg) {
  const yaw = yawDeg * Math.PI / 180;
  const along = { x: Math.cos(yaw), y: Math.sin(yaw) };
  const across = { x: -along.y, y: along.x };
  const halfLength = 0.43;
  const halfWidth = 0.105;
  const levels = [0.05, 0.18];
  const corners = [
    [-halfLength, -halfWidth],
    [halfLength, -halfWidth],
    [halfLength, halfWidth],
    [-halfLength, halfWidth],
  ];
  return levels.flatMap((z) => corners.map(([u, v]) => ({
    x: along.x * u + across.x * v,
    y: along.y * u + across.y * v,
    z,
  })));
}

function drawCuboid(ctx, frame, yawDeg, style) {
  const vertices = cuboidVertices(yawDeg).map((point) => projectPoint(frame, point));
  const faces = [
    [0, 1, 2, 3],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
  ].sort((a, b) => {
    const ay = a.reduce((sum, index) => sum + vertices[index].y, 0) / a.length;
    const by = b.reduce((sum, index) => sum + vertices[index].y, 0) / b.length;
    return ay - by;
  });
  ctx.save();
  ctx.setLineDash(style.dash || []);
  for (const face of faces) {
    ctx.beginPath();
    face.forEach((index, offset) => {
      const point = vertices[index];
      if (offset === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.fillStyle = style.fill;
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = style.lineWidth;
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function prepareCanvas(canvas, background = '#ffffff') {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 3);
  const pixelWidth = Math.max(1, Math.round(rect.width * pixelRatio));
  const pixelHeight = Math.max(1, Math.round(rect.height * pixelRatio));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, rect.width, rect.height);
  return { ctx, rect, frame: poseFrame(rect.width, rect.height) };
}

function drawCoordinateBase(ctx, frame) {
  const gridValues = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75];
  for (const value of gridValues) {
    line(
      ctx,
      projectPoint(frame, { x: -0.75, y: value, z: 0 }),
      projectPoint(frame, { x: 0.75, y: value, z: 0 }),
      value === 0 ? '#d5dade' : '#edf0f2',
      value === 0 ? 1 : 0.75,
    );
    line(
      ctx,
      projectPoint(frame, { x: value, y: -0.75, z: 0 }),
      projectPoint(frame, { x: value, y: 0.75, z: 0 }),
      value === 0 ? '#d5dade' : '#edf0f2',
      value === 0 ? 1 : 0.75,
    );
  }

  drawAxis(ctx, frame, { x: 0.86, y: 0, z: 0 }, '#ef5350', 'X');
  drawAxis(ctx, frame, { x: 0, y: 0.86, z: 0 }, '#43a047', 'Y');
  drawAxis(ctx, frame, { x: 0, y: 0, z: 0.65 }, '#3b82f6', 'Z');
}

function drawPoseWindow() {
  const prepared = prepareCanvas(elements.poseCanvas);
  if (!prepared) return;
  const { ctx, frame } = prepared;
  drawCoordinateBase(ctx, frame);

  const actualYaw = latestState?.actual?.yaw_deg ?? -90;
  const targetYaw = previewYawDeg ?? latestState?.target?.yaw_deg ?? -90;
  drawCuboid(ctx, frame, actualYaw, {
    fill: 'rgba(78, 85, 91, 0.48)',
    stroke: 'rgba(47, 54, 59, 0.90)',
    lineWidth: 1.1,
  });
  drawCuboid(ctx, frame, targetYaw, {
    fill: 'rgba(44, 190, 232, 0.18)',
    stroke: 'rgba(17, 139, 174, 0.92)',
    lineWidth: 1.25,
    dash: [4, 2],
  });
}

function recordControllerSample(state) {
  if (!state?.wrench_local || !Number.isFinite(state.time)) return;
  const previous = controllerHistory[controllerHistory.length - 1];
  if (previous && state.time < previous.time) controllerHistory = [];
  const latest = controllerHistory[controllerHistory.length - 1];
  if (!latest || state.time - latest.time >= CONTROLLER_SAMPLE_INTERVAL_SECONDS * 0.9) {
    controllerHistory.push({
      time: state.time,
      lateral: Math.abs(state.wrench_local.fy),
      height: state.wrench_local.fz,
      torque: state.wrench_local.tz,
    });
  }
  const cutoff = state.time - CONTROLLER_HISTORY_SECONDS;
  controllerHistory = controllerHistory
    .filter((sample) => sample.time >= cutoff)
    .slice(-96);
}

function drawHistoryPlot(canvas, accessor, options) {
  const prepared = prepareCanvas(canvas, '#071018');
  if (!prepared) return;
  const { ctx, rect } = prepared;
  const margin = { left: 32, right: 7, top: 5, bottom: 15 };
  const plot = {
    left: margin.left,
    right: rect.width - margin.right,
    top: margin.top,
    bottom: rect.height - margin.bottom,
  };
  const latestTime = latestState?.time || 0;
  const startTime = latestTime - CONTROLLER_HISTORY_SECONDS;
  const observedMaximum = Math.max(
    0,
    ...controllerHistory.map((sample) => Math.abs(accessor(sample))),
  );
  const span = Math.min(options.limit, Math.max(options.minimumSpan, observedMaximum * 1.18));
  const minimum = options.signed ? -span : 0;
  const maximum = span;
  const x = (time) => plot.left + (time - startTime) / CONTROLLER_HISTORY_SECONDS * (plot.right - plot.left);
  const y = (value) => plot.bottom - (value - minimum) / (maximum - minimum) * (plot.bottom - plot.top);

  ctx.font = '8px DM Mono, monospace';
  ctx.fillStyle = '#78909c';
  ctx.textAlign = 'right';
  const ticks = options.signed ? [-span, 0, span] : [0, span * 0.5, span];
  for (const value of ticks) {
    const screenY = y(value);
    line(ctx, { x: plot.left, y: screenY }, { x: plot.right, y: screenY }, value === 0 ? 'rgba(117, 185, 211, .28)' : 'rgba(117, 185, 211, .10)', 1);
    ctx.fillText(`${value > 0 && options.signed ? '+' : ''}${value.toFixed(options.decimals || 0)}`, plot.left - 4, screenY + 3);
  }
  ctx.textAlign = 'left';
  ctx.fillText('−3 s', plot.left, rect.height - 4);
  ctx.textAlign = 'right';
  ctx.fillText('now', plot.right, rect.height - 4);

  const visible = controllerHistory.filter((sample) => sample.time >= startTime);
  if (visible.length === 0) return;
  ctx.save();
  ctx.strokeStyle = options.color;
  ctx.shadowColor = options.glow;
  ctx.shadowBlur = 7;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  visible.forEach((sample, index) => {
    const pointX = x(sample.time);
    const value = Math.max(minimum, Math.min(maximum, accessor(sample)));
    const pointY = y(value);
    if (index === 0) ctx.moveTo(pointX, pointY);
    else ctx.lineTo(pointX, pointY);
  });
  ctx.stroke();
  ctx.restore();
}

function drawControllerHistories() {
  drawHistoryPlot(elements.lateralForceCanvas, (sample) => sample.lateral, {
    color: '#d946ef',
    glow: 'rgba(217, 70, 239, .72)',
    signed: false,
    minimumSpan: 5,
    limit: latestState?.controller?.horizontal_force_limit || 100,
  });
  drawHistoryPlot(elements.heightForceCanvas, (sample) => sample.height, {
    color: '#45d8ff',
    glow: 'rgba(69, 216, 255, .70)',
    signed: true,
    minimumSpan: 20,
    limit: latestState?.controller?.height_force_limit || 300,
  });
  drawHistoryPlot(elements.torqueHistoryCanvas, (sample) => sample.torque, {
    color: '#f97316',
    glow: 'rgba(249, 115, 22, .72)',
    signed: true,
    minimumSpan: 0.5,
    limit: latestState?.controller?.yaw_torque_limit || 10,
    decimals: 1,
  });
}

function renderState(state) {
  latestState = state;
  if (state.ready === false) {
    elements.connection.classList.remove('online');
    elements.connection.classList.toggle('error', Boolean(state.runtime_error));
    elements.connection.querySelector('span').textContent = state.runtime_error ? 'Setup failed' : 'Preparing runtime';
    elements.loading.classList.remove('hidden');
    elements.loading.classList.toggle('is-error', Boolean(state.runtime_error));
    elements.loading.querySelector('[data-loading-prefix]').textContent = state.runtime_error ? 'MuJoCo setup failed' : 'Preparing';
    elements.loading.querySelector('[data-loading-suffix]').textContent = state.runtime_error ? '' : 'COLA in Mujoco';
    elements.loading.querySelector('.cola-icon').hidden = Boolean(state.runtime_error);
    elements.loading.querySelector('small').textContent = state.loading_message || 'Loading the model, controllers, and student policy';
    return;
  }
  elements.connection.classList.remove('error');
  elements.connection.classList.add('online');
  elements.connection.querySelector('span').textContent = state.healthy ? 'Simulation healthy' : 'Simulation unstable';
  elements.loading.classList.toggle('hidden', !state.runtime_error);
  if (state.runtime_error) {
    elements.loading.classList.add('is-error');
    elements.loading.querySelector('[data-loading-prefix]').textContent = 'Simulation paused';
    elements.loading.querySelector('[data-loading-suffix]').textContent = '';
    elements.loading.querySelector('.cola-icon').hidden = true;
    elements.loading.querySelector('small').textContent = state.runtime_error;
  }

  setText('#simulation-time', `t = ${number(state.time, 2)} s`);
  setText('#base-height', `${number(state.base.height, 3)} m`);
  setText('#actual-height', `${number(state.actual.height, 3)} m`);
  setText('#actual-speed', `${number(Math.hypot(state.actual.vx, state.actual.vy), 3)} m/s`);
  const yawError = normalizeDegrees(state.target.yaw_deg - state.actual.yaw_deg);
  setText('#yaw-error', `${number(Math.abs(yawError), 1)}°`);
  setText('#velocity-force-value', `|Fy| ${number(Math.abs(state.wrench_local.fy), 1)} N`);
  setText('#height-force-value', `Fz ${number(state.wrench_local.fz, 1, true)} N`);
  setText('#torque-value', `τz ${number(state.wrench_local.tz, 2, true)} N·m`);
  recordControllerSample(state);
  document.querySelectorAll('[data-overlay]').forEach((button) => {
    button.setAttribute('aria-pressed', String(Boolean(state.overlays?.[button.dataset.overlay])));
  });

  if (!activeSliders.has(elements.height.id)) elements.height.value = state.target.height;
  if (velocityPointerId === null) {
    setText('#target-vx', number(state.target.vx, 2, true));
    setText('#target-vy', number(state.target.vy, 2, true));
    updateVelocityThumb(state.target.vx, state.target.vy);
  }
  setText('#target-height', `${number(Number(elements.height.value), 2)} m`);
  setText('#target-yaw', degrees(previewYawDeg ?? state.target.yaw_deg));
  setText('#physics-rate', `${number(state.rates.physics_hz, 0)} Hz`);
  setText('#controller-rate', `${number(state.rates.controller_hz, 0)} Hz`);
  setText('#policy-rate', `${number(state.rates.policy_hz, 0)} Hz`);

  elements.pause.querySelector('.pause-icon').textContent = state.paused ? '▶' : 'Ⅱ';
  elements.pause.querySelector('.pause-label').textContent = state.paused ? 'resume' : 'pause';
  elements.pause.title = state.paused ? 'Resume simulation' : 'Pause simulation';
  elements.pause.setAttribute('aria-label', elements.pause.title);
  elements.cameraFollow.classList.toggle('active', state.camera_follow);
  elements.cameraFollow.querySelector('.camera-follow-icon').textContent = state.camera_follow ? '◎' : '○';
  elements.cameraFollow.title = state.camera_follow ? 'Disable camera follow' : 'Enable camera follow';
  elements.cameraFollow.setAttribute('aria-label', elements.cameraFollow.title);
  drawPoseWindow();
  drawControllerHistories();
}

async function invokeRuntime(operation, payload = {}) {
  if (!browserRuntime) throw new Error('Browser runtime is not initialized');
  if (operation === 'state') return browserRuntime.getState();
  await runtimePromise;
  if (operation === 'command') return browserRuntime.command(payload.command);
  if (operation === 'velocity') return browserRuntime.setVelocity(payload.vx, payload.vy);
  if (operation === 'height') return browserRuntime.setHeight(payload.height);
  if (operation === 'yaw') return browserRuntime.setYawDegrees(payload.yawDeg);
  if (operation === 'controller-overlay') return browserRuntime.toggleControllerOverlay(payload.controller);
  if (operation === 'camera-follow') return browserRuntime.toggleCameraFollow();
  if (operation === 'default-targets') return browserRuntime.resetTargets();
  throw new Error(`Unsupported browser-runtime operation: ${operation}`);
}

async function sendCommand(command, sourceButton = null) {
  if (sourceButton) {
    sourceButton.classList.add('pressed');
    window.setTimeout(() => sourceButton.classList.remove('pressed'), 100);
  }
  try {
    previewYawDeg = null;
    renderState(await invokeRuntime('command', { command }));
  } catch (error) {
    showToast(error.message);
  }
}

function scheduleVelocityFlush() {
  if (velocityFlushTimer !== null || velocityRequestPending || !queuedVelocityTarget) return;
  velocityFlushTimer = window.setTimeout(() => {
    velocityFlushTimer = null;
    flushVelocityTarget();
  }, 32);
}

async function flushVelocityTarget() {
  if (velocityRequestPending || !queuedVelocityTarget) return;
  velocityRequestPending = true;
  const target = queuedVelocityTarget;
  queuedVelocityTarget = null;
  try {
    renderState(await invokeRuntime('velocity', target));
  } catch (error) {
    showToast(error.message);
  } finally {
    velocityRequestPending = false;
    scheduleVelocityFlush();
  }
}

function queueVelocityTarget(vx, vy) {
  queuedVelocityTarget = { vx, vy };
  scheduleVelocityFlush();
}

function scheduleHeightFlush() {
  if (heightFlushTimer !== null || heightRequestPending || queuedHeightTarget === null) return;
  heightFlushTimer = window.setTimeout(() => {
    heightFlushTimer = null;
    flushHeightTarget();
  }, 32);
}

async function flushHeightTarget() {
  if (heightRequestPending || queuedHeightTarget === null) return;
  heightRequestPending = true;
  const height = queuedHeightTarget;
  queuedHeightTarget = null;
  try {
    renderState(await invokeRuntime('height', { height }));
  } catch (error) {
    showToast(error.message);
  } finally {
    heightRequestPending = false;
    scheduleHeightFlush();
  }
}

function queueHeightTarget() {
  queuedHeightTarget = Number(elements.height.value);
  scheduleHeightFlush();
}

function scheduleYawFlush() {
  if (yawFlushTimer !== null || yawRequestPending || queuedYawTarget === null) return;
  yawFlushTimer = window.setTimeout(() => {
    yawFlushTimer = null;
    flushYawTarget();
  }, 32);
}

async function flushYawTarget() {
  if (yawRequestPending || queuedYawTarget === null) return;
  yawRequestPending = true;
  const yawDeg = queuedYawTarget;
  queuedYawTarget = null;
  try {
    const state = await invokeRuntime('yaw', { yawDeg });
    if (posePointerId === null && queuedYawTarget === null) previewYawDeg = null;
    renderState(state);
  } catch (error) {
    showToast(error.message);
  } finally {
    yawRequestPending = false;
    scheduleYawFlush();
  }
}

function queueYawTarget(yawDeg) {
  previewYawDeg = normalizeDegrees(yawDeg);
  queuedYawTarget = previewYawDeg;
  setText('#target-yaw', degrees(previewYawDeg));
  drawPoseWindow();
  scheduleYawFlush();
}

function bindSlider(element, onInput) {
  const activate = () => activeSliders.add(element.id);
  const deactivate = () => activeSliders.delete(element.id);
  element.addEventListener('pointerdown', activate);
  element.addEventListener('pointerup', deactivate);
  element.addEventListener('pointercancel', deactivate);
  element.addEventListener('blur', deactivate);
  element.addEventListener('input', onInput);
  element.addEventListener('change', deactivate);
}

bindSlider(elements.height, () => {
  setText('#target-height', `${number(Number(elements.height.value), 2)} m`);
  queueHeightTarget();
});

function velocityFromPointer(event) {
  const rect = elements.velocityTrack.getBoundingClientRect();
  const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  const limits = velocityLimits();
  return {
    vx: Number((limits.vx[1] - y * (limits.vx[1] - limits.vx[0])).toFixed(3)),
    vy: Number((limits.vy[1] - x * (limits.vy[1] - limits.vy[0])).toFixed(3)),
  };
}

function applyVelocityPointer(event) {
  const target = velocityFromPointer(event);
  updateVelocityThumb(target.vx, target.vy);
  setText('#target-vx', number(target.vx, 2, true));
  setText('#target-vy', number(target.vy, 2, true));
  queueVelocityTarget(target.vx, target.vy);
}

elements.velocityTrack.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  velocityPointerId = event.pointerId;
  elements.velocityTrack.setPointerCapture(event.pointerId);
  elements.velocityTrack.classList.add('dragging');
  applyVelocityPointer(event);
});
elements.velocityTrack.addEventListener('pointermove', (event) => {
  if (event.pointerId === velocityPointerId) applyVelocityPointer(event);
});
elements.velocityTrack.addEventListener('pointerup', (event) => {
  if (event.pointerId !== velocityPointerId) return;
  applyVelocityPointer(event);
  velocityPointerId = null;
  elements.velocityTrack.classList.remove('dragging');
});
elements.velocityTrack.addEventListener('pointercancel', (event) => {
  if (event.pointerId !== velocityPointerId) return;
  velocityPointerId = null;
  elements.velocityTrack.classList.remove('dragging');
});
elements.velocityTrack.addEventListener('dblclick', (event) => {
  event.preventDefault();
  updateVelocityThumb(0, 0);
  setText('#target-vx', number(0, 2, true));
  setText('#target-vy', number(0, 2, true));
  queueVelocityTarget(0, 0);
});

function yawFromPointer(event) {
  const rect = elements.poseCanvas.getBoundingClientRect();
  const frame = poseFrame(rect.width, rect.height);
  const dx = event.clientX - rect.left - frame.origin.x;
  const dy = event.clientY - rect.top - frame.origin.y;
  const determinant = frame.ex.x * frame.ey.y - frame.ey.x * frame.ex.y;
  if (Math.abs(determinant) < 1e-6) return previewYawDeg ?? latestState?.target?.yaw_deg ?? -90;
  const x = (dx * frame.ey.y - frame.ey.x * dy) / determinant;
  const y = (frame.ex.x * dy - dx * frame.ex.y) / determinant;
  if (Math.hypot(x, y) < 0.08) return previewYawDeg ?? latestState?.target?.yaw_deg ?? -90;
  return Math.atan2(y, x) * 180 / Math.PI;
}

function applyPosePointer(event) {
  queueYawTarget(yawFromPointer(event));
}

elements.poseCanvas.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  posePointerId = event.pointerId;
  elements.poseCanvas.setPointerCapture(event.pointerId);
  applyPosePointer(event);
});
elements.poseCanvas.addEventListener('pointermove', (event) => {
  if (event.pointerId === posePointerId) applyPosePointer(event);
});
elements.poseCanvas.addEventListener('pointerup', (event) => {
  if (event.pointerId !== posePointerId) return;
  applyPosePointer(event);
  posePointerId = null;
});
elements.poseCanvas.addEventListener('pointercancel', (event) => {
  if (event.pointerId === posePointerId) posePointerId = null;
});

document.querySelectorAll('[data-command]').forEach((button) => {
  button.addEventListener('click', () => sendCommand(button.dataset.command, button));
});

document.querySelectorAll('[data-overlay]').forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      renderState(await invokeRuntime('controller-overlay', {
        controller: button.dataset.overlay,
      }));
    } catch (error) {
      showToast(error.message);
    }
  });
});

const keyboardCommands = new Set(['w', 'a', 's', 'd', 'i', 'k', 'j', 'l', 'c', 'x', 'r']);
window.addEventListener('keydown', (event) => {
  if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
  const activeTag = document.activeElement?.tagName;
  const activeInputType = document.activeElement?.getAttribute?.('type');
  if (activeTag === 'TEXTAREA' || activeTag === 'SELECT' || (activeTag === 'INPUT' && activeInputType !== 'range')) return;
  const command = event.key.toLowerCase();
  if (!keyboardCommands.has(command)) return;
  event.preventDefault();
  const button = document.querySelector(`[data-command="${command}"]`);
  sendCommand(command, button);
});

elements.cameraFollow.addEventListener('click', async () => {
  try { renderState(await invokeRuntime('camera-follow')); }
  catch (error) { showToast(error.message); }
});

elements.dock.addEventListener('pointerdown', (event) => event.stopPropagation());
elements.dock.addEventListener('wheel', (event) => event.stopPropagation(), { passive: true });
elements.controllerVisuals.addEventListener('pointerdown', (event) => event.stopPropagation());
elements.controllerVisuals.addEventListener('wheel', (event) => event.stopPropagation(), { passive: true });

const visualizationObserver = new ResizeObserver(() => {
  drawPoseWindow();
  drawControllerHistories();
});
for (const canvas of [
  elements.poseCanvas,
  elements.lateralForceCanvas,
  elements.heightForceCanvas,
  elements.torqueHistoryCanvas,
]) visualizationObserver.observe(canvas);

async function pollState() {
  try {
    renderState(await invokeRuntime('state'));
  } catch (error) {
    elements.connection.classList.remove('online');
    elements.connection.classList.add('error');
    elements.connection.querySelector('span').textContent = 'Runtime unavailable';
  } finally {
    window.setTimeout(pollState, 40);
  }
}

async function initializeDemo() {
  if (!window.crossOriginIsolated) {
    renderState({
      ready: false,
      runtime_error: null,
      loading_message: 'Enabling the secure browser simulation runtime; this page will reload once',
    });
    return;
  }
  browserRuntime = new ColaBrowserRuntime(elements.stream, ({ message }) => {
    renderState({ ready: false, loading_message: message, runtime_error: null });
  });
  window.addEventListener('beforeunload', () => browserRuntime.dispose(), { once: true });
  runtimePromise = browserRuntime.initialize({ start: !SMOKE_MODE });
  try {
    renderState(await invokeRuntime('default-targets'));
    if (SMOKE_MODE) {
      for (let step = 0; step < 240; step += 1) await browserRuntime.runPhysicsStep();
      browserRuntime.renderer.render();
      const state = browserRuntime.getState();
      renderState(state);
      document.body.dataset.smoke = state.healthy ? 'pass' : 'fail';
      document.body.dataset.simulationTime = state.time.toFixed(3);
      console.log('COLA_DEMO_SMOKE_PASS', JSON.stringify(state));
      return;
    }
  } catch (error) {
    showToast(error.message);
    if (SMOKE_MODE) document.body.dataset.smoke = 'fail';
  } finally {
    if (SMOKE_MODE) return;
    pollState();
  }
}

drawPoseWindow();
drawControllerHistories();
initializeDemo();
