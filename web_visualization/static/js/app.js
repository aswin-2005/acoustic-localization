import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ==========================================
// SOCKET + SHARED STATE
// ==========================================
const socket = io();

let roomSize = { w: 10, l: 10, h: 4 };
let micPos   = new THREE.Vector3(5, 2, 5); // Three.js: Y=up, Z=depth
let lastTime = performance.now();
let lastEventTime = performance.now();

const THEME = {
    bg:   0x1e2736,
    grid: 0x4a5462,
    mic:  0x58a6ff,
    cam:  0x39d2c0,
    gt:   0xf85149,
    pred: 0x3fb950,
};

// ==========================================
// SCENE
// ==========================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(THEME.bg);
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(5, 12, 8);
scene.add(dirLight);

// Container groups – rebuilt on config change
const roomGroup = new THREE.Group();
scene.add(roomGroup);

let gtMesh, predMesh;

// ==========================================
// CAMERA SYSTEM
// Encapsulates the 3D model + virtual POV lens.
//
// Three.js lookAt() convention:
//   The object's LOCAL +Z axis is pointed AWAY from the target.
//   i.e.: if you call obj.lookAt(target), the object's local -Z
//   faces the target (that is the Three.js "forward" direction).
//
// We build the physical camera model with the lens on the -Z side.
// That way lookAt(soundTarget) makes the lens face the sound — correct.
//
// The virtual POV camera is placed 0.1m BEHIND the rig origin
// (in the +Z local direction, which is the camera's own "back"),
// so it looks outward through the lens.  We manually position it
// using the world-space direction and negate it.
// ==========================================
class CameraSystem {
    constructor() {
        this.rig = new THREE.Group();
        this.model = new THREE.Group(); // camera body, built lens-on-(-Z)
        this.pov = new THREE.PerspectiveCamera(60, 1, 0.05, 200);

        // Logical azimuth/elevation (degrees, initialised facing -X = Az 180)
        this.az    = 180;
        this.el    = 0;
        this.tgtAz = 180;
        this.tgtEl = 0;
        this.speed = 45; // deg/s

        this._buildModel();
        this.rig.add(this.model);
        scene.add(this.rig);
    }

    // Build a realistic CCTV-camera shape.
    // Lens parts are at NEGATIVE Z so that after lookAt() the lens
    // points toward the target.
    _buildModel() {
        // --- body ---
        const body = new THREE.Mesh(
            new THREE.BoxGeometry(0.30, 0.22, 0.38),
            new THREE.MeshPhongMaterial({ color: 0x2e3035 })
        );
        this.model.add(body);

        // --- lens barrel (projects toward +Z) ---
        // Object3D.lookAt() makes LOCAL +Z face the target (opposite of Camera).
        // So the lens must be on the +Z side to face the sound source.
        const barrel = new THREE.Mesh(
            new THREE.CylinderGeometry(0.09, 0.09, 0.20, 32),
            new THREE.MeshPhongMaterial({ color: 0x111111 })
        );
        barrel.rotation.x = Math.PI / 2;
        barrel.position.z = 0.25;    // ← lens side is +Z
        this.model.add(barrel);

        // --- lens glass ---
        const glass = new THREE.Mesh(
            new THREE.CylinderGeometry(0.07, 0.07, 0.02, 32),
            new THREE.MeshPhongMaterial({ color: 0x0055cc, shininess: 120, opacity: 0.85, transparent: true })
        );
        glass.rotation.x = Math.PI / 2;
        glass.position.z = 0.35;     // ← front of lens
        this.model.add(glass);

        // --- top accent stripe ---
        const top = new THREE.Mesh(
            new THREE.BoxGeometry(0.14, 0.05, 0.18),
            new THREE.MeshPhongMaterial({ color: THEME.cam })
        );
        top.position.set(0, 0.135, 0);
        this.model.add(top);

        // Lens is now on +Z. Object3D.lookAt() makes +Z face the target.
        this.model.rotation.set(0, 0, 0);
    }

    setPosition(pos) {
        this.rig.position.copy(pos);
    }

    setTarget(az, el) {
        this.tgtAz = az;
        this.tgtEl = el;
    }

    update(dt) {
        // --- smooth interpolation ---
        const dAz  = shortestAngle(this.tgtAz, this.az);
        const dEl  = this.tgtEl - this.el;
        const step = this.speed * dt;
        if (Math.abs(dAz) > 0.05) this.az += Math.sign(dAz) * Math.min(step, Math.abs(dAz));
        if (Math.abs(dEl) > 0.05) this.el += Math.sign(dEl) * Math.min(step, Math.abs(dEl));
        this.az = ((this.az + 180) % 360) - 180;

        // --- world-space "look direction" from this rig ---
        const dir    = getWorldDir(this.az, this.el);
        const target = new THREE.Vector3().copy(this.rig.position).addScaledVector(dir, 1.0);

        // Object3D.lookAt() makes the local +Z axis face the target (NOT -Z — that's Camera behaviour).
        // Our lens is on +Z, so rig.lookAt(target) makes the lens face the target correctly.
        this.rig.lookAt(target);

        // POV camera is a THREE Camera object — Camera.lookAt() makes -Z face the target.
        // The lens glass is at +0.35 m from the rig origin in the +Z (forward) direction.
        // The POV must sit just BEYOND the lens (>= 0.35m ahead) so the camera body
        // geometry doesn't clip into the near-plane and appear as a black circle.
        this.pov.position.copy(this.rig.position).addScaledVector(dir, 0.4);
        this.pov.lookAt(target);

        // UI
        const azDisp = ((this.az % 360) + 360) % 360;
        document.getElementById("camera-stats").innerText =
            `Az: ${azDisp.toFixed(1)}°  El: ${this.el.toFixed(1)}°`;
    }

    dispose() {
        scene.remove(this.rig);
    }
}

// ==========================================
// HELPERS
// ==========================================
function degToRad(d) { return d * Math.PI / 180; }

function shortestAngle(target, current) {
    let d = (target - current) % 360;
    return d > 180 ? d - 360 : d < -180 ? d + 360 : d;
}

// Az=0 → +X, Az=90 → +Z, Az=180 → -X  (standard physics convention)
// Three.js: Y=up
function getWorldDir(azDeg, elDeg) {
    const az = degToRad(azDeg);
    const el = degToRad(elDeg);
    return new THREE.Vector3(
        Math.cos(el) * Math.cos(az),
        Math.sin(el),
        Math.cos(el) * Math.sin(az)
    ).normalize();
}

// ==========================================
// ROOM BUILDER
// ==========================================
let camSystem = null;

function buildRoom() {
    // clear previous room geometry
    while (roomGroup.children.length) roomGroup.remove(roomGroup.children[0]);
    // remove old markers from scene
    if (gtMesh)   scene.remove(gtMesh);
    if (predMesh) scene.remove(predMesh);

    const { w, h, l } = roomSize;

    // Wireframe box
    const boxWire = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, l)),
        new THREE.LineBasicMaterial({ color: THEME.grid, transparent: true, opacity: 0.55 })
    );
    boxWire.position.set(w/2, h/2, l/2);
    roomGroup.add(boxWire);

    // Floor grid
    const grid = new THREE.GridHelper(Math.max(w, l), 20, THEME.grid, THEME.grid);
    grid.position.set(w/2, 0, l/2);
    roomGroup.add(grid);

    // Axis sticks: X=red, Z=green, Y(up)=blue
    const axisMat = (c) => new THREE.MeshBasicMaterial({ color: c });
    const stick = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);

    const ax = new THREE.Mesh(stick, axisMat(0xff2222));
    ax.rotation.z = -Math.PI / 2;
    ax.position.set(w / 2, 0, 0);
    ax.scale.y = w;
    roomGroup.add(ax);

    const az_ = new THREE.Mesh(stick, axisMat(0x22ff22));
    az_.rotation.x = Math.PI / 2;
    az_.position.set(0, 0, l / 2);
    az_.scale.y = l;
    roomGroup.add(az_);

    const ay = new THREE.Mesh(stick, axisMat(0x2222ff));
    ay.position.set(0, h / 2, 0);
    ay.scale.y = h;
    roomGroup.add(ay);

    // Mic tetrahedron
    const micGroup = new THREE.Group();
    micGroup.position.copy(micPos).add(new THREE.Vector3(0, 0.35, 0));
    micGroup.add(new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 16, 16),
        new THREE.MeshPhongMaterial({ color: 0x222222 })
    ));
    const micD = 0.13;
    const micVerts = [
        new THREE.Vector3( micD, 0, -micD / Math.sqrt(2)),
        new THREE.Vector3(-micD, 0, -micD / Math.sqrt(2)),
        new THREE.Vector3(0,  micD,  micD / Math.sqrt(2)),
        new THREE.Vector3(0, -micD,  micD / Math.sqrt(2)),
    ];
    const capsule = new THREE.SphereGeometry(0.03, 12, 12);
    const capMat  = new THREE.MeshPhongMaterial({ color: THEME.mic });
    const lineMat = new THREE.LineBasicMaterial({ color: 0x888888 });
    micVerts.forEach(v => {
        const n = new THREE.Mesh(capsule, capMat);
        n.position.copy(v);
        micGroup.add(n);
        micGroup.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), v]),
            lineMat
        ));
    });
    micGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -0.35, 0), new THREE.Vector3()]),
        lineMat
    ));
    roomGroup.add(micGroup);

    // Event markers
    gtMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.15, 16, 16),
        new THREE.MeshPhongMaterial({ color: THEME.gt, emissive: THEME.gt, emissiveIntensity: 0.6 })
    );
    gtMesh.visible = false;
    scene.add(gtMesh);

    predMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.2, 0.2, 0.2),
        new THREE.MeshPhongMaterial({ color: THEME.pred, emissive: THEME.pred, emissiveIntensity: 0.6 })
    );
    predMesh.visible = false;
    scene.add(predMesh);

    // Camera system: rebuild once per config
    if (camSystem) camSystem.dispose();
    camSystem = new CameraSystem();
    camSystem.setPosition(micPos);
}

// ==========================================
// RENDERERS + VIEWPORT CAMERAS
// ==========================================
const container3D = document.getElementById("view3d-container");
const renderer3D  = new THREE.WebGLRenderer({ antialias: true });
renderer3D.setPixelRatio(window.devicePixelRatio);
renderer3D.setSize(container3D.clientWidth, container3D.clientHeight);
container3D.appendChild(renderer3D.domElement);

const camera3D = new THREE.PerspectiveCamera(
    45, container3D.clientWidth / container3D.clientHeight, 0.1, 500
);
camera3D.position.set(-4, roomSize.h * 2, -4);
camera3D.lookAt(roomSize.w / 2, roomSize.h / 2, roomSize.l / 2);

const controls3D = new OrbitControls(camera3D, renderer3D.domElement);
controls3D.target.set(roomSize.w / 2, roomSize.h / 2, roomSize.l / 2);
controls3D.update();

const containerPOV = document.getElementById("viewpov-container");
const rendererPOV  = new THREE.WebGLRenderer({ antialias: true });
rendererPOV.setPixelRatio(window.devicePixelRatio);
rendererPOV.setSize(containerPOV.clientWidth, containerPOV.clientHeight);
containerPOV.appendChild(rendererPOV.domElement);

// Crosshair overlay
const crosshair = document.createElement("div");
crosshair.style.cssText = "position:absolute;top:50%;left:50%;width:20px;height:20px;transform:translate(-50%,-50%);pointer-events:none";
crosshair.innerHTML = '<div style="width:100%;height:2px;background:#39d2c0;position:absolute;top:9px"></div><div style="width:2px;height:100%;background:#39d2c0;position:absolute;left:9px"></div>';
containerPOV.appendChild(crosshair);

// Axis labels (HTML overlay)
const labels = {};
function makeLabel(text, color) {
    const el = document.createElement("div");
    el.style.cssText = `position:absolute;color:${color};font-weight:bold;font-size:12px;pointer-events:none;`;
    el.innerText = text;
    container3D.appendChild(el);
    return el;
}
labels.X = makeLabel("+X", "#ff4444");
labels.Z = makeLabel("+Z", "#44ff44");
labels.Y = makeLabel("+Y (Up)", "#4444ff");

window.addEventListener("resize", () => {
    camera3D.aspect = container3D.clientWidth / container3D.clientHeight;
    camera3D.updateProjectionMatrix();
    renderer3D.setSize(container3D.clientWidth, container3D.clientHeight);

    if (camSystem) {
        camSystem.pov.aspect = containerPOV.clientWidth / containerPOV.clientHeight;
        camSystem.pov.updateProjectionMatrix();
    }
    rendererPOV.setSize(containerPOV.clientWidth, containerPOV.clientHeight);
});

// ==========================================
// ANIMATION LOOP
// ==========================================
function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    const dt  = Math.min((now - lastTime) / 1000, 0.1); // cap to avoid jumps
    lastTime  = now;

    if (!camSystem) { renderer3D.render(scene, camera3D); return; }

    // Auto-return home after 5 s of silence
    if (now - lastEventTime > 5000) {
        camSystem.setTarget(180, 0);
    }

    camSystem.update(dt);

    // Pulse GT marker
    if (gtMesh && gtMesh.visible) {
        const s = 1.0 + 0.18 * Math.sin(now / 200);
        gtMesh.scale.setScalar(s);
    }
    // Spin prediction marker
    if (predMesh && predMesh.visible) {
        predMesh.rotation.y += 1.8 * dt;
        predMesh.rotation.x += 0.9 * dt;
    }

    // Update axis labels (projected to screen)
    const hw = container3D.clientWidth  / 2;
    const hh = container3D.clientHeight / 2;
    const project = (v3, lbl) => {
        const p = v3.clone().project(camera3D);
        lbl.style.left = (p.x * hw + hw) + "px";
        lbl.style.top  = (-p.y * hh + hh) + "px";
    };
    project(new THREE.Vector3(roomSize.w + 0.3, 0, 0),         labels.X);
    project(new THREE.Vector3(0, 0, roomSize.l + 0.3),         labels.Z);
    project(new THREE.Vector3(0, roomSize.h + 0.3, 0),         labels.Y);

    controls3D.update();
    renderer3D.render(scene, camera3D);
    rendererPOV.render(scene, camSystem.pov);
}
animate();

// ==========================================
// CONFIG — REST (reload-resilient) + Socket
// ==========================================
function applyConfig(data) {
    let changed = false;
    if (data.room_w && data.room_l && data.room_h) {
        if (roomSize.w !== data.room_w || roomSize.l !== data.room_l || roomSize.h !== data.room_h) {
            roomSize = { w: data.room_w, l: data.room_l, h: data.room_h };
            changed = true;
        }
        document.getElementById("room-stats").innerText =
            `${roomSize.w} × ${roomSize.l} × ${roomSize.h} m  |  Mic @ [${data.mic_pos.join(", ")}]`;
    }
    if (data.mic_pos) {
        // Python pipeline: mic_pos = [x, y, z] where z is height
        // Three.js:        Y = height, Z = depth
        const mp = data.mic_pos;
        const nx = mp[0], ny = mp[2], nz = mp[1];
        if (micPos.x !== nx || micPos.y !== ny || micPos.z !== nz) {
            micPos.set(nx, ny, nz);
            changed = true;
        }
    }
    if (!camSystem || changed) {
        buildRoom();
        controls3D.target.set(roomSize.w / 2, roomSize.h / 2, roomSize.l / 2);
        camera3D.position.set(-4, roomSize.h * 2, -4);
        camera3D.lookAt(controls3D.target);
        controls3D.update();
    }
}

function loadConfig() {
    fetch("/api/config")
        .then(r => r.json())
        .then(data => { console.log("[Config] REST", data); applyConfig(data); })
        .catch(err => {
            console.warn("[Config] REST failed, using defaults:", err);
            if (!camSystem) buildRoom(); // still render something
        });
}
window.addEventListener("load", loadConfig);

// ==========================================
// SOCKET EVENTS
// ==========================================
socket.on("connect", () => {
    document.querySelector(".status-indicator").style.backgroundColor = "#" + THEME.pred.toString(16).padStart(6, "0");
    document.getElementById("status-text").innerText = "Connected";
});
socket.on("disconnect", () => {
    document.querySelector(".status-indicator").style.backgroundColor = "#" + THEME.gt.toString(16).padStart(6, "0");
    document.getElementById("status-text").innerText = "Disconnected";
});
socket.on("init_data", (data) => {
    console.log("[Config] Socket init_data", data);
    applyConfig(data);
});

socket.on("ping_data", (data) => {
    const logBox = document.getElementById("log-content");

    if (data.event_type && data.event_type !== "Silence") {
        lastEventTime = performance.now();

        // Ground-truth marker
        if (data.true_azimuth != null && data.true_elevation != null) {
            gtMesh.visible = true;
            const d = getWorldDir(data.true_azimuth, data.true_elevation);
            gtMesh.position.copy(micPos).addScaledVector(d, 2.5);
        }

        // Prediction marker + camera target
        if (data.pred_azimuth != null && data.pred_elevation != null) {
            predMesh.visible = true;
            const d = getWorldDir(data.pred_azimuth, data.pred_elevation);
            predMesh.position.copy(micPos).addScaledVector(d, 2.5);

            // Camera follows prediction direction directly — no artificial offset
            camSystem.setTarget(data.pred_azimuth, data.pred_elevation);
        }

        const msg = document.createElement("div");
        msg.className = "log-entry log-hit";
        msg.innerText = `[EVENT] ${data.event_type}  GT: ${data.true_azimuth?.toFixed(1)}° / ${data.true_elevation?.toFixed(1)}°`;
        logBox.appendChild(msg);

        if (data.pred_azimuth != null) {
            const pm = document.createElement("div");
            pm.className = "log-entry log-pred";
            pm.innerText = `[PRED]  Az: ${data.pred_azimuth.toFixed(1)}°  El: ${data.pred_elevation.toFixed(1)}°`;
            logBox.appendChild(pm);
        }
    } else {
        gtMesh.visible = false;
        predMesh.visible = false;
        const msg = document.createElement("div");
        msg.className = "log-entry log-info";
        msg.innerText = "[IDLE] Monitoring…";
        logBox.appendChild(msg);
    }

    logBox.scrollTop = logBox.scrollHeight;
    while (logBox.children.length > 50) logBox.removeChild(logBox.firstChild);
});
