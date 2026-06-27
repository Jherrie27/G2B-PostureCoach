import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const MODEL_URL = "../models/posture_lgbm_v3.txt";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const CLASSES = ["correct_posture", "slouching", "neck_forward", "lean"];
const FEATURE_ORDER = [
  "ear_shoulder_offset_x",
  "craniovertebral_angle",
  "head_forward_offset_z",
  "nose_shoulder_offset_x",
  "shoulder_roll_z",
  "torso_compression_ratio",
  "elbow_forward_offset_z",
  "spine_angle_3d",
  "shoulder_tilt_angle",
  "hip_tilt_angle",
  "midline_deviation_angle",
  "nose_centerline_offset_x",
  "lateral_asymmetry_index",
  "landmark_confidence_mean",
];

const LANDMARK_INDICES = {
  nose: 0,
  left_ear: 7,
  right_ear: 8,
  left_shoulder: 11,
  right_shoulder: 12,
  left_elbow: 13,
  right_elbow: 14,
  left_hip: 23,
  right_hip: 24,
};
const ORDERED_NAMES = Object.keys(LANDMARK_INDICES);
const IDX = Object.fromEntries(ORDERED_NAMES.map((name, index) => [name, index]));
const NORMAL_THRESHOLDS = {
  ear_shoulder_offset_x: 0.2,
  shoulder_roll_z: 0.08,
  torso_compression_min: 1.55,
  shoulder_tilt_abs_max: 3,
  midline_deviation_max: 3,
  craniovertebral_max: 12,
};

const elements = {
  startCamera: document.querySelector("#startCamera"),
  stopCamera: document.querySelector("#stopCamera"),
  switchCamera: document.querySelector("#switchCamera"),
  cameraStatus: document.querySelector("#cameraStatus"),
  emptyState: document.querySelector("#emptyState"),
  video: document.querySelector("#video"),
  overlay: document.querySelector("#overlay"),
  postureClass: document.querySelector("#postureClass"),
  confidenceValue: document.querySelector("#confidenceValue"),
  probabilityList: document.querySelector("#probabilityList"),
  viewQuality: document.querySelector("#viewQuality"),
  forwardHeadMetric: document.querySelector("#forwardHeadMetric"),
  shoulderRollMetric: document.querySelector("#shoulderRollMetric"),
  tiltMetric: document.querySelector("#tiltMetric"),
  sessionMetric: document.querySelector("#sessionMetric"),
  coachFeed: document.querySelector("#coachFeed"),
  coachForm: document.querySelector("#coachForm"),
  coachInput: document.querySelector("#coachInput"),
};

let poseLandmarker = null;
let drawingUtils = null;
let lgbmModel = null;
let stream = null;
let running = false;
let autoStartTried = false;
let mirrored = true;
let lastVideoTime = -1;
let animationId = null;
let lastState = null;
let smoother = new TemporalSmoother(0.3, 8);
let sessionTracker = new SessionTracker();

init();

async function init() {
  elements.startCamera.disabled = true;
  renderProbabilityBars(CLASSES.map(() => 0));
  addAssistantMessage("Start camera when ready.");
  wireEvents();

  try {
    setStatus("Loading model");
    const [vision, model] = await Promise.all([
      FilesetResolver.forVisionTasks(WASM_URL),
      loadLightGbmModel(MODEL_URL),
    ]);
    lgbmModel = model;
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    drawingUtils = new DrawingUtils(elements.overlay.getContext("2d"));
    setStatus("Ready", "ok");
    elements.startCamera.disabled = false;
    startCamera({ auto: true });
  } catch (error) {
    console.error(error);
    setStatus("Model load failed", "error");
    elements.startCamera.disabled = true;
    addAssistantMessage("The browser could not load the posture model. Refresh the page and check the network connection.");
  }
}

function wireEvents() {
  elements.startCamera.addEventListener("click", () => startCamera());
  elements.stopCamera.addEventListener("click", () => stopCamera());
  elements.switchCamera.addEventListener("click", () => {
    mirrored = !mirrored;
    applyMirrorState();
  });
  elements.coachForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitCoachQuestion(elements.coachInput.value);
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => submitCoachQuestion(button.dataset.prompt));
  });
}

async function startCamera(options = {}) {
  const auto = Boolean(options.auto);
  if (auto && autoStartTried) {
    return;
  }
  if (auto) {
    autoStartTried = true;
  }

  if (!poseLandmarker) {
    setStatus("Still loading", "warn");
    elements.startCamera.disabled = true;
    return;
  }

  stopCamera();
  resetSession();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
    elements.video.srcObject = stream;
    await elements.video.play();
    applyMirrorState();
    elements.emptyState.classList.add("hidden");
    elements.startCamera.disabled = true;
    elements.stopCamera.disabled = false;
    elements.switchCamera.disabled = false;
    running = true;
    setStatus("Camera running", "ok");
    animationId = requestAnimationFrame(predictLoop);
  } catch (error) {
    console.error(error);
    setStatus(auto ? "Camera permission needed" : "Camera blocked", auto ? "warn" : "error");
    elements.startCamera.disabled = false;
    elements.stopCamera.disabled = true;
    elements.switchCamera.disabled = true;
    addAssistantMessage(
      auto
        ? "The browser did not start the webcam automatically. Press Start camera and allow camera access."
        : "Camera permission was blocked or unavailable."
    );
  }
}

function stopCamera() {
  running = false;
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  clearCanvas();
  elements.video.srcObject = null;
  elements.emptyState.classList.remove("hidden");
  elements.startCamera.disabled = !poseLandmarker;
  elements.stopCamera.disabled = true;
  elements.switchCamera.disabled = true;
  if (poseLandmarker) {
    setStatus("Ready", "ok");
  }
}

function resetSession() {
  lastState = null;
  lastVideoTime = -1;
  smoother = new TemporalSmoother(0.3, 8);
  sessionTracker = new SessionTracker();
  elements.postureClass.className = "posture-class";
  elements.postureClass.textContent = "Waiting";
  elements.confidenceValue.textContent = "--";
  elements.viewQuality.textContent = "No reading";
  elements.viewQuality.className = "quality-badge";
  elements.forwardHeadMetric.textContent = "--";
  elements.shoulderRollMetric.textContent = "--";
  elements.tiltMetric.textContent = "--";
  elements.sessionMetric.textContent = "0:00";
  renderProbabilityBars(CLASSES.map(() => 0));
}

function applyMirrorState() {
  elements.video.classList.toggle("is-mirrored", mirrored);
  elements.overlay.classList.toggle("is-mirrored", mirrored);
}

function predictLoop() {
  if (!running) {
    return;
  }

  const video = elements.video;
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    resizeOverlay();
    const result = poseLandmarker.detectForVideo(video, performance.now());
    drawPose(result);
    updateFromPose(result);
  }

  animationId = requestAnimationFrame(predictLoop);
}

function resizeOverlay() {
  const video = elements.video;
  if (video.videoWidth && video.videoHeight) {
    if (elements.overlay.width !== video.videoWidth) {
      elements.overlay.width = video.videoWidth;
    }
    if (elements.overlay.height !== video.videoHeight) {
      elements.overlay.height = video.videoHeight;
    }
  }
}

function drawPose(result) {
  clearCanvas();
  if (!result.landmarks || result.landmarks.length === 0 || !drawingUtils) {
    return;
  }

  const landmarks = result.landmarks[0];
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#0f766e",
    lineWidth: 4,
  });
  drawingUtils.drawLandmarks(landmarks, {
    color: "#ffffff",
    fillColor: "#b91c1c",
    lineWidth: 2,
    radius: 4,
  });
}

function clearCanvas() {
  const ctx = elements.overlay.getContext("2d");
  ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
}

function updateFromPose(result) {
  if (!result.landmarks || result.landmarks.length === 0) {
    setStatus("No person detected", "warn");
    setQuality("No reading", "warn");
    return;
  }

  const rows = pickLandmarks(result.landmarks[0]);
  const quality = landmarkQuality(rows);
  const normalized = normalizeLandmarks(rows);
  const features = normalized ? extractFeatures(normalized) : null;

  if (!features || !quality.upperReliable) {
    setStatus("Adjust framing", "warn");
    setQuality("Partial view", "warn");
    return;
  }

  const prediction = predictPosture(features);
  const smoothed = smoother.update(prediction.probs);
  const session = sessionTracker.update(smoothed.label);
  const confidence = smoothed.probs[CLASSES.indexOf(smoothed.label)];

  lastState = {
    postureClass: smoothed.label,
    confidence,
    classProbabilities: Object.fromEntries(CLASSES.map((label, index) => [label, smoothed.probs[index]])),
    features,
    quality,
    session,
    primaryIssue: primaryIssue(features),
  };

  setStatus("Camera running", "ok");
  setQuality(quality.fullReliable ? "Full view" : "Partial view", quality.fullReliable ? "ok" : "warn");
  renderState(lastState);
}

function pickLandmarks(landmarks) {
  return ORDERED_NAMES.map((name) => {
    const lm = landmarks[LANDMARK_INDICES[name]];
    return {
      x: lm.x,
      y: lm.y,
      z: lm.z,
      visibility: typeof lm.visibility === "number" ? lm.visibility : 1,
    };
  });
}

function landmarkQuality(rows) {
  const mean = rows.reduce((sum, row) => sum + row.visibility, 0) / rows.length;
  const upperReliable = [
    IDX.left_ear,
    IDX.right_ear,
    IDX.left_shoulder,
    IDX.right_shoulder,
  ].every((index) => rows[index].visibility >= 0.35);
  const fullReliable = [
    IDX.left_ear,
    IDX.right_ear,
    IDX.left_shoulder,
    IDX.right_shoulder,
    IDX.left_hip,
    IDX.right_hip,
  ].every((index) => rows[index].visibility >= 0.5);
  return { mean, upperReliable, fullReliable };
}

function normalizeLandmarks(rows) {
  const leftHip = rows[IDX.left_hip];
  const rightHip = rows[IDX.right_hip];
  const leftShoulder = rows[IDX.left_shoulder];
  const rightShoulder = rows[IDX.right_shoulder];
  const hipMid = midpoint(leftHip, rightHip);
  const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  if (!Number.isFinite(shoulderWidth) || shoulderWidth < 0.02) {
    return null;
  }
  return rows.map((row) => ({
    x: (row.x - hipMid.x) / shoulderWidth,
    y: (row.y - hipMid.y) / shoulderWidth,
    z: (row.z - hipMid.z) / shoulderWidth,
    visibility: row.visibility,
  }));
}

function extractFeatures(n) {
  const nose = n[IDX.nose];
  const le = n[IDX.left_ear];
  const re = n[IDX.right_ear];
  const ls = n[IDX.left_shoulder];
  const rs = n[IDX.right_shoulder];
  const lel = n[IDX.left_elbow];
  const rel = n[IDX.right_elbow];
  const lh = n[IDX.left_hip];
  const rh = n[IDX.right_hip];
  const earMid = midpoint(le, re);
  const shMid = midpoint(ls, rs);
  const hipMid = midpoint(lh, rh);
  const elbowMid = midpoint(lel, rel);

  const values = {
    ear_shoulder_offset_x: earMid.z - shMid.z,
    craniovertebral_angle: deviationFromVerticalDeg({ x: 0, y: earMid.y - shMid.y, z: earMid.z - shMid.z }),
    head_forward_offset_z: earMid.z - shMid.z,
    nose_shoulder_offset_x: nose.x - shMid.x,
    shoulder_roll_z: shMid.z - hipMid.z,
    torso_compression_ratio: Math.abs(shMid.y - hipMid.y),
    elbow_forward_offset_z: elbowMid.z - shMid.z,
    spine_angle_3d: deviationFromVerticalDeg(shMid),
    shoulder_tilt_angle: lineTiltFromHorizontalDeg(ls, rs),
    hip_tilt_angle: lineTiltFromHorizontalDeg(lh, rh),
    midline_deviation_angle: deviationFromVerticalDeg({ x: shMid.x - hipMid.x, y: shMid.y - hipMid.y, z: 0 }),
    nose_centerline_offset_x: nose.x - shMid.x,
    lateral_asymmetry_index: distance(le, ls) - distance(re, rs),
    landmark_confidence_mean: n.reduce((sum, row) => sum + row.visibility, 0) / n.length,
  };

  return Object.values(values).every(Number.isFinite) ? values : null;
}

function predictPosture(features) {
  const vector = FEATURE_ORDER.map((name) => features[name]);
  const mlProbs = lgbmModel ? lgbmModel.predict(vector) : null;
  const rules = ruleProbs(features);
  const probs = mlProbs
    ? normalizeProbs(mlProbs.map((prob, index) => 0.7 * prob + 0.3 * rules[index]))
    : rules;
  const bestIndex = argmax(probs);
  return {
    label: CLASSES[bestIndex],
    confidence: probs[bestIndex],
    probs,
  };
}

async function loadLightGbmModel(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load LightGBM model: ${response.status}`);
  }
  return parseLightGbmModel(await response.text());
}

function parseLightGbmModel(text) {
  const header = {};
  const trees = [];
  let current = null;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    if (line.startsWith("Tree=")) {
      if (current) {
        trees.push(finalizeTree(current));
      }
      current = { index: Number(line.slice(5)) };
      continue;
    }

    const eq = line.indexOf("=");
    if (eq === -1) {
      continue;
    }
    const key = line.slice(0, eq);
    const value = line.slice(eq + 1);

    if (!current) {
      header[key] = value;
      continue;
    }

    if (key === "num_leaves") {
      current.numLeaves = Number(value);
    } else if (key === "split_feature") {
      current.splitFeature = parseNumberArray(value);
    } else if (key === "threshold") {
      current.threshold = parseNumberArray(value);
    } else if (key === "left_child") {
      current.leftChild = parseNumberArray(value);
    } else if (key === "right_child") {
      current.rightChild = parseNumberArray(value);
    } else if (key === "leaf_value") {
      current.leafValue = parseNumberArray(value);
    }
  }

  if (current) {
    trees.push(finalizeTree(current));
  }

  return {
    numClass: Number(header.num_class || 4),
    numTreePerIteration: Number(header.num_tree_per_iteration || 4),
    trees,
    predict(features) {
      const rawScores = Array(this.numClass).fill(0);
      this.trees.forEach((tree, index) => {
        rawScores[index % this.numTreePerIteration] += evalTree(tree, features);
      });
      return softmax(rawScores);
    },
  };
}

function finalizeTree(tree) {
  const required = ["splitFeature", "threshold", "leftChild", "rightChild", "leafValue"];
  for (const key of required) {
    if (!tree[key]) {
      tree[key] = [];
    }
  }
  return tree;
}

function parseNumberArray(value) {
  const trimmed = value.trim();
  return trimmed ? trimmed.split(/\s+/).map(Number) : [];
}

function evalTree(tree, features) {
  let node = 0;
  let guard = 0;
  while (guard < 128) {
    guard += 1;
    const featureIndex = tree.splitFeature[node];
    const threshold = tree.threshold[node];
    const featureValue = features[featureIndex];
    const child = featureValue <= threshold ? tree.leftChild[node] : tree.rightChild[node];
    if (child < 0) {
      return tree.leafValue[-child - 1] || 0;
    }
    node = child;
  }
  return 0;
}

function ruleProbs(f) {
  const leanScore = Math.max(
    sigmoid((Math.abs(f.shoulder_tilt_angle) - 4) / 2),
    sigmoid((f.midline_deviation_angle - 4) / 2),
  );
  const neckScore = Math.max(
    sigmoid((f.ear_shoulder_offset_x - 0.25) / 0.1),
    sigmoid((f.craniovertebral_angle - 12) / 4),
  );
  const slouchScore = Math.max(
    sigmoid((f.shoulder_roll_z - 0.1) / 0.05),
    sigmoid((1.45 - f.torso_compression_ratio) / 0.1),
  );
  const correctScore = Math.max(1 - Math.max(leanScore, neckScore, slouchScore), 0.01);
  return normalizeProbs([correctScore, slouchScore, neckScore, leanScore]);
}

function renderState(state) {
  const label = state.postureClass;
  elements.postureClass.className = `posture-class ${label}`;
  elements.postureClass.textContent = formatLabel(label);
  elements.confidenceValue.textContent = `${Math.round(state.confidence * 100)}%`;
  elements.forwardHeadMetric.textContent = signed(state.features.ear_shoulder_offset_x, 2);
  elements.shoulderRollMetric.textContent = signed(state.features.shoulder_roll_z, 2);
  elements.tiltMetric.textContent = `${signed(state.features.shoulder_tilt_angle, 1)} deg`;
  elements.sessionMetric.textContent = formatDuration(state.session.sessionDurationSec);
  renderProbabilityBars(CLASSES.map((className) => state.classProbabilities[className] || 0));
}

function renderProbabilityBars(probs) {
  elements.probabilityList.innerHTML = "";
  CLASSES.forEach((className, index) => {
    const item = document.createElement("div");
    item.className = "probability-item";
    item.innerHTML = `
      <span>${formatLabel(className)}</span>
      <div class="probability-track"><div class="probability-fill"></div></div>
      <strong>${Math.round((probs[index] || 0) * 100)}%</strong>
    `;
    item.querySelector(".probability-fill").style.width = `${Math.round((probs[index] || 0) * 100)}%`;
    elements.probabilityList.appendChild(item);
  });
}

function setStatus(text, tone = "") {
  elements.cameraStatus.textContent = text;
  elements.cameraStatus.className = `status-pill ${tone}`.trim();
}

function setQuality(text, tone = "") {
  elements.viewQuality.textContent = text;
  elements.viewQuality.className = `quality-badge ${tone}`.trim();
}

function submitCoachQuestion(rawQuestion) {
  const question = (rawQuestion || "").trim();
  if (!question) {
    return;
  }
  addUserMessage(question);
  addAssistantMessage(makeCoachResponse(question, lastState));
  elements.coachInput.value = "";
}

function addAssistantMessage(text) {
  addMessage("assistant", text);
}

function addUserMessage(text) {
  addMessage("user", text);
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  elements.coachFeed.appendChild(message);
  elements.coachFeed.scrollTop = elements.coachFeed.scrollHeight;
}

function makeCoachResponse(question, state) {
  if (!state) {
    return "I do not have a posture reading yet. Start the camera and keep your head and shoulders visible.";
  }

  const text = question.toLowerCase();
  const label = state.postureClass;
  const issue = state.primaryIssue;
  const feature = state.features;
  const classText = formatLabel(label).toLowerCase();

  if (text.includes("what") && text.includes("posture")) {
    return `Your current posture reads as ${classText} with ${Math.round(state.confidence * 100)}% confidence. The strongest signal is ${formatIssue(issue)}.`;
  }

  if (text.includes("setup") || text.includes("chair") || text.includes("desk") || text.includes("monitor")) {
    return setupAdvice(label, feature);
  }

  if (text.includes("exercise") || text.includes("stretch") || text.includes("correct") || text.includes("fix") || text.includes("improve")) {
    return correctionAdvice(label, feature);
  }

  if (label === "correct_posture") {
    return "You are currently in a good position. Keep your ears stacked over your shoulders, relax your jaw, and take a short movement break before fatigue changes your posture.";
  }

  return correctionAdvice(label, feature);
}

function correctionAdvice(label, feature) {
  if (label === "neck_forward") {
    return `Your head is drifting forward: offset ${signed(feature.ear_shoulder_offset_x, 2)}, craniovertebral angle ${feature.craniovertebral_angle.toFixed(0)} deg. Bring the screen closer to eye level, tuck your chin gently, and stack your ears over your shoulders.`;
  }
  if (label === "slouching") {
    return `Your shoulder roll is elevated at ${signed(feature.shoulder_roll_z, 2)}. Lift your sternum slightly, let the shoulder blades settle back and down, and support the lower back instead of rounding forward.`;
  }
  if (label === "lean") {
    return `Your shoulder tilt is ${signed(feature.shoulder_tilt_angle, 1)} deg. Put both feet flat, center your weight on both sitting bones, and bring the keyboard and monitor back to the midline.`;
  }
  return "Your posture is currently reading as correct. Keep the same alignment and reset your shoulders if you feel yourself rounding forward.";
}

function setupAdvice(label, feature) {
  if (label === "neck_forward") {
    return `Raise the display or lower the chair until the top third of the screen is near eye level. The current forward-head offset is ${signed(feature.ear_shoulder_offset_x, 2)}, so also bring the keyboard and mouse closer.`;
  }
  if (label === "slouching") {
    return `Use the chair back or a small lumbar support so your torso does not collapse. Your torso compression is ${feature.torso_compression_ratio.toFixed(2)}, and higher is better for this measurement.`;
  }
  if (label === "lean") {
    return `Your setup may be pulling you sideways. Center the monitor, keep the mouse close to your dominant side, and check that both armrests are at the same height.`;
  }
  return "The setup looks workable right now. Keep the monitor centered, feet flat, elbows near your sides, and the keyboard close enough that your shoulders stay relaxed.";
}

function primaryIssue(features) {
  const deviations = {
    forward_head: Math.max(0, features.ear_shoulder_offset_x - NORMAL_THRESHOLDS.ear_shoulder_offset_x),
    shoulder_roll: Math.max(0, features.shoulder_roll_z - NORMAL_THRESHOLDS.shoulder_roll_z),
    torso_compression: Math.max(0, NORMAL_THRESHOLDS.torso_compression_min - features.torso_compression_ratio),
    shoulder_tilt: Math.max(0, Math.abs(features.shoulder_tilt_angle) - NORMAL_THRESHOLDS.shoulder_tilt_abs_max),
    midline_deviation: Math.max(0, Math.abs(features.midline_deviation_angle) - NORMAL_THRESHOLDS.midline_deviation_max),
  };
  let best = "none";
  let bestValue = 0;
  for (const [key, value] of Object.entries(deviations)) {
    if (value > bestValue) {
      best = key;
      bestValue = value;
    }
  }
  return best;
}

class TemporalSmoother {
  constructor(alpha, hysteresisFrames) {
    this.alpha = alpha;
    this.hysteresis = hysteresisFrames;
    this.smoothed = null;
    this.currentLabel = "correct_posture";
    this.candidateLabel = null;
    this.candidateStreak = 0;
  }

  update(rawProbs) {
    if (!this.smoothed) {
      this.smoothed = [...rawProbs];
    } else {
      this.smoothed = rawProbs.map((prob, index) => this.alpha * prob + (1 - this.alpha) * this.smoothed[index]);
      this.smoothed = normalizeProbs(this.smoothed);
    }

    const argmaxLabel = CLASSES[argmax(this.smoothed)];
    if (argmaxLabel === this.currentLabel) {
      this.candidateLabel = null;
      this.candidateStreak = 0;
    } else if (argmaxLabel === this.candidateLabel) {
      this.candidateStreak += 1;
    } else {
      this.candidateLabel = argmaxLabel;
      this.candidateStreak = 1;
    }

    if (this.candidateStreak >= this.hysteresis) {
      this.currentLabel = argmaxLabel;
      this.candidateLabel = null;
      this.candidateStreak = 0;
    }

    return {
      label: this.currentLabel,
      probs: [...this.smoothed],
    };
  }
}

class SessionTracker {
  constructor() {
    this.sessionStart = performance.now();
    this.currentLabel = null;
    this.currentLabelStart = this.sessionStart;
    this.timePerClass = Object.fromEntries(CLASSES.map((label) => [label, 0]));
    this.correctionEvents = 0;
    this.longestBadStreak = 0;
    this.lastBadStreakStart = null;
  }

  update(label) {
    const now = performance.now();
    if (!this.currentLabel) {
      this.currentLabel = label;
      this.currentLabelStart = now;
      if (label !== "correct_posture") {
        this.lastBadStreakStart = now;
      }
      return this.snapshot(now, label);
    }

    if (label !== this.currentLabel) {
      this.timePerClass[this.currentLabel] += (now - this.currentLabelStart) / 1000;
      if (label === "correct_posture" && this.currentLabel !== "correct_posture") {
        this.correctionEvents += 1;
        if (this.lastBadStreakStart !== null) {
          this.longestBadStreak = Math.max(this.longestBadStreak, (now - this.lastBadStreakStart) / 1000);
          this.lastBadStreakStart = null;
        }
      }
      if (label !== "correct_posture" && this.currentLabel === "correct_posture") {
        this.lastBadStreakStart = now;
      }
      this.currentLabel = label;
      this.currentLabelStart = now;
    }

    return this.snapshot(now, label);
  }

  snapshot(now, label) {
    const live = { ...this.timePerClass };
    live[label] += (now - this.currentLabelStart) / 1000;
    const total = Object.values(live).reduce((sum, value) => sum + value, 0) || 1;
    return {
      postureDurationSec: (now - this.currentLabelStart) / 1000,
      sessionDurationSec: (now - this.sessionStart) / 1000,
      postureDistribution: Object.fromEntries(CLASSES.map((className) => [className, live[className] / total])),
      correctionEvents: this.correctionEvents,
      longestBadPostureStreakSec: this.longestBadStreak,
    };
  }
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function deviationFromVerticalDeg(vec) {
  const norm = Math.hypot(vec.x, vec.y, vec.z) || 1e-9;
  const cosang = clamp(vec.y / norm, -1, 1);
  const angle = radiansToDegrees(Math.acos(cosang));
  return Math.min(angle, 180 - angle);
}

function lineTiltFromHorizontalDeg(from, to) {
  const angle = radiansToDegrees(Math.atan2(to.y - from.y, to.x - from.x));
  if (angle > 90) {
    return angle - 180;
  }
  if (angle <= -90) {
    return angle + 180;
  }
  return angle;
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - maxValue));
  const total = exp.reduce((sum, value) => sum + value, 0) || 1;
  return exp.map((value) => value / total);
}

function normalizeProbs(values) {
  const safe = values.map((value) => (Number.isFinite(value) && value > 0 ? value : 0));
  const total = safe.reduce((sum, value) => sum + value, 0) || 1;
  return safe.map((value) => value / total);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function argmax(values) {
  return values.reduce((bestIndex, value, index, arr) => (value > arr[bestIndex] ? index : bestIndex), 0);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function radiansToDegrees(value) {
  return value * 180 / Math.PI;
}

function formatLabel(label) {
  return label
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatIssue(issue) {
  if (issue === "none") {
    return "no major deviation";
  }
  return issue.replaceAll("_", " ");
}

function signed(value, digits) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const remainder = `${whole % 60}`.padStart(2, "0");
  return `${minutes}:${remainder}`;
}
