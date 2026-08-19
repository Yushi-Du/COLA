import * as THREE from 'three';
import { OrbitControls } from './vendor/three/addons/controls/OrbitControls.js';


function colorAndOpacity(model, geomId) {
  const materialId = model.geom_matid[geomId];
  const source = materialId >= 0 ? model.mat_rgba : model.geom_rgba;
  const offset = 4 * (materialId >= 0 ? materialId : geomId);
  return {
    color: new THREE.Color(source[offset], source[offset + 1], source[offset + 2]),
    opacity: source[offset + 3],
  };
}


export class MujocoRenderer {
  constructor(canvas, mujoco, model, data, ids) {
    this.canvas = canvas;
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.ids = ids;
    this.follow = true;
    this.lastFollowTarget = new THREE.Vector3(0.1, 0, 0.78);
    this.geometryCache = new Map();
    this.renderGeoms = [];

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x071016);
    this.scene.fog = new THREE.FogExp2(0x071016, 0.026);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.03, 200);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.65, -2.65, 1.72);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 3));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.target.copy(this.lastFollowTarget);
    this.controls.minDistance = 1.25;
    this.controls.maxDistance = 8;
    this.controls.minPolarAngle = 0.20;
    this.controls.maxPolarAngle = 0.52 * Math.PI;

    this.addLights();
    this.buildGeometries();
    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
  }

  addLights() {
    this.scene.add(new THREE.HemisphereLight(0xdceeff, 0x172028, 1.65));
    const key = new THREE.DirectionalLight(0xffffff, 3.0);
    key.position.set(2.8, -2.4, 5.5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -4;
    key.shadow.camera.right = 4;
    key.shadow.camera.top = 4;
    key.shadow.camera.bottom = -4;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 12;
    key.shadow.bias = -0.0002;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x6acfff, 1.25);
    rim.position.set(-3, 2, 2.5);
    this.scene.add(rim);
  }

  shouldRenderGeom(geomId) {
    const alpha = this.model.geom_rgba[4 * geomId + 3];
    if (alpha <= 0.001) return false;
    const type = this.model.geom_type[geomId];
    if (type !== this.mujoco.mjtGeom.mjGEOM_MESH.value) return true;
    if (this.model.geom_group[geomId] !== 0) return true;
    const bodyId = this.model.geom_bodyid[geomId];
    const dataId = this.model.geom_dataid[geomId];
    for (let other = 0; other < this.model.ngeom; other += 1) {
      if (
        other !== geomId
        && this.model.geom_type[other] === type
        && this.model.geom_group[other] === 1
        && this.model.geom_bodyid[other] === bodyId
        && this.model.geom_dataid[other] === dataId
      ) return false;
    }
    return true;
  }

  meshGeometry(meshId) {
    const key = `mesh:${meshId}`;
    if (this.geometryCache.has(key)) return this.geometryCache.get(key);
    const vertexAddress = this.model.mesh_vertadr[meshId];
    const vertexCount = this.model.mesh_vertnum[meshId];
    const faceAddress = this.model.mesh_faceadr[meshId];
    const faceCount = this.model.mesh_facenum[meshId];
    const positions = new Float32Array(3 * vertexCount);
    for (let i = 0; i < positions.length; i += 1) {
      positions[i] = this.model.mesh_vert[3 * vertexAddress + i];
    }
    const IndexArray = vertexCount > 65535 ? Uint32Array : Uint16Array;
    const indices = new IndexArray(3 * faceCount);
    for (let i = 0; i < indices.length; i += 1) {
      indices[i] = this.model.mesh_face[3 * faceAddress + i];
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    this.geometryCache.set(key, geometry);
    return geometry;
  }

  primitiveGeometry(geomId) {
    const type = this.model.geom_type[geomId];
    const size = Array.from(this.model.geom_size.subarray(3 * geomId, 3 * geomId + 3));
    const key = `primitive:${type}:${size.join(':')}`;
    if (this.geometryCache.has(key)) return this.geometryCache.get(key);
    let geometry;
    if (type === this.mujoco.mjtGeom.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(200, 200, 1, 1);
    } else if (type === this.mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0], 24, 16);
    } else if (type === this.mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(size[0], 2 * size[2], 8, 16);
      geometry.rotateX(0.5 * Math.PI);
    } else if (type === this.mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1, 24, 16);
      geometry.scale(size[0], size[1], size[2]);
    } else if (type === this.mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], 2 * size[2], 24);
      geometry.rotateX(0.5 * Math.PI);
    } else if (type === this.mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    } else {
      geometry = new THREE.BufferGeometry();
      console.warn(`Unsupported MuJoCo geom type ${type}`);
    }
    this.geometryCache.set(key, geometry);
    return geometry;
  }

  buildGeometries() {
    for (let geomId = 0; geomId < this.model.ngeom; geomId += 1) {
      if (!this.shouldRenderGeom(geomId)) continue;
      const type = this.model.geom_type[geomId];
      const geometry = type === this.mujoco.mjtGeom.mjGEOM_MESH.value
        ? this.meshGeometry(this.model.geom_dataid[geomId])
        : this.primitiveGeometry(geomId);
      const appearance = colorAndOpacity(this.model, geomId);
      const isGround = type === this.mujoco.mjtGeom.mjGEOM_PLANE.value;
      if (isGround) appearance.color.set(0x202a31);
      const material = new THREE.MeshStandardMaterial({
        color: appearance.color,
        roughness: isGround ? 0.94 : 0.58,
        metalness: isGround ? 0 : 0.04,
        transparent: appearance.opacity < 0.995,
        opacity: appearance.opacity,
        depthWrite: appearance.opacity >= 0.8,
        side: isGround ? THREE.DoubleSide : THREE.FrontSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = !isGround && appearance.opacity > 0.8;
      mesh.receiveShadow = true;
      mesh.renderOrder = appearance.opacity < 0.995 ? 2 : 0;
      this.scene.add(mesh);
      this.renderGeoms.push({ geomId, mesh });

      if (isGround) {
        const grid = new THREE.GridHelper(200, 400, 0x526571, 0x35434b);
        grid.rotation.x = 0.5 * Math.PI;
        grid.position.z = 0.0015;
        grid.material.transparent = true;
        grid.material.opacity = 0.35;
        grid.material.depthWrite = false;
        this.scene.add(grid);
      }
    }
  }

  updateTransforms() {
    for (const { geomId, mesh } of this.renderGeoms) {
      const positionOffset = 3 * geomId;
      const matrixOffset = 9 * geomId;
      const matrix = this.data.geom_xmat;
      mesh.matrix.set(
        matrix[matrixOffset], matrix[matrixOffset + 1], matrix[matrixOffset + 2], this.data.geom_xpos[positionOffset],
        matrix[matrixOffset + 3], matrix[matrixOffset + 4], matrix[matrixOffset + 5], this.data.geom_xpos[positionOffset + 1],
        matrix[matrixOffset + 6], matrix[matrixOffset + 7], matrix[matrixOffset + 8], this.data.geom_xpos[positionOffset + 2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
    }
  }

  updateCameraFollow() {
    if (!this.follow) return;
    const address = this.ids.baseQposAddress;
    const nextTarget = new THREE.Vector3(
      this.data.qpos[address] + 0.10,
      this.data.qpos[address + 1],
      0.78,
    );
    const delta = nextTarget.clone().sub(this.lastFollowTarget);
    this.camera.position.add(delta);
    this.controls.target.copy(nextTarget);
    this.lastFollowTarget.copy(nextTarget);
  }

  cameraState() {
    const offset = this.camera.position.clone().sub(this.controls.target);
    const horizontal = Math.hypot(offset.x, offset.y);
    return {
      azimuth: Math.atan2(offset.x, -offset.y) * 180 / Math.PI,
      elevation: -Math.atan2(offset.z, horizontal) * 180 / Math.PI,
      distance: offset.length(),
    };
  }

  resize() {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 3);
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.updateTransforms();
    this.updateCameraFollow();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.resizeObserver?.disconnect();
    this.controls.dispose();
    for (const { mesh } of this.renderGeoms) mesh.material.dispose();
    for (const geometry of this.geometryCache.values()) geometry.dispose();
    this.renderer.dispose();
  }
}
