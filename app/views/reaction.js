import { css, esc } from "../ui.js";
import { createLatestFrameScheduler, supportsImage } from "./reaction-scheduler.js";

const FRAME_WIDTH = 320;
const FRAME_HEIGHT = 240;
const MIN_FRAME_INTERVAL_MS = 200;
const STATE = "Classify this camera frame.";

const BASE_QUESTIONS = {
  person_present: {
    type: "noul",
    instructions: "Does the frame visibly contain at least one person?",
  },
  smiling: {
    type: "noul",
    instructions: "Does a visible person have a smiling or visibly happy facial expression? Judge only the visible expression, not inner emotion.",
  },
};

const EXPRESSION_QUESTIONS = {
  neutral: {
    type: "noul",
    instructions: "Does a visible person show a neutral facial expression? Judge only visible facial cues.",
  },
  surprised: {
    type: "noul",
    instructions: "Does a visible person show a surprised facial expression? Judge only visible facial cues.",
  },
  sad: {
    type: "noul",
    instructions: "Does a visible person show a sad facial expression? Judge only visible facial cues.",
  },
};

const LABELS = {
  person_present: "Person visibly present",
  smiling: "Smiling / visibly happy expression",
  neutral: "Neutral expression",
  surprised: "Surprised expression",
  sad: "Sad expression",
};

const TEMPLATE = `<div class="wrap">
  <div class="page-head">
    <div class="eyebrow">Demo · local camera analysis</div>
    <h1>Reaction Recorder</h1>
    <p>See how a decision model scores visible people and facial expressions in a live camera frame. Scores describe visible cues only, not a person's inner mood.</p>
  </div>

  <section class="rr-gate card pad" data-f="gate" aria-live="polite">
    <div class="rr-gate-copy">
      <span class="eyebrow">Image model required</span>
      <h2 data-f="gate-title">Choose an image-capable model</h2>
      <p class="muted" data-f="gate-detail"></p>
    </div>
    <button class="btn primary" type="button" data-act="open-picker">Open model picker</button>
  </section>

  <section class="rr-app" data-f="app" hidden>
    <div class="rr-toolbar card pad">
      <div class="rr-live-state">
        <span class="rr-status-dot" data-f="status-dot" aria-hidden="true"></span>
        <div><b data-f="camera-status">Camera is off</b><span class="tiny muted" data-f="camera-detail">Start only when you are ready.</span></div>
      </div>
      <label class="rr-extra-toggle"><input type="checkbox" data-f="extra-questions"> Include neutral, surprised, and sad expression scores <span class="tiny faint">adds three questions per frame</span></label>
      <div class="rr-actions">
        <button class="btn primary" type="button" data-act="start">Start camera</button>
        <button class="btn ghost" type="button" data-act="pause" hidden>Pause camera</button>
        <button class="btn ghost" type="button" data-act="stop" hidden>Stop camera</button>
        <button class="btn ghost" type="button" data-act="cancel" hidden>Cancel</button>
      </div>
    </div>

    <div class="rr-grid">
      <div class="rr-video-card card">
        <div class="rr-video-frame" data-f="video-frame">
          <video data-f="video" muted playsinline aria-label="Live camera preview"></video>
          <div class="rr-placeholder" data-f="placeholder"><span class="rr-camera-icon" aria-hidden="true">◉</span><b>Camera is off</b><span>Press Start camera when ready.</span></div>
          <span class="rr-frame-label">320 × 240 · frames processed on this device</span>
        </div>
      </div>

      <aside class="rr-results card pad">
        <div class="rr-results-head"><div><span class="eyebrow">Live model scores · uncalibrated</span><h2>Visible cues</h2></div><span class="rr-model-tag" data-f="model-label"></span></div>
        <div class="rr-metrics">
          <div><span>Processed FPS</span><b data-f="fps">—</b></div>
          <div><span>Frame latency</span><b data-f="latency">—</b></div>
        </div>
        <div class="rr-scores" data-f="scores" aria-live="off"><p class="muted small">Start the camera to see model scores.</p></div>
        <p class="rr-caveat">Model scores are uncalibrated estimates. Facial expressions do not reliably reveal a person's actual mood.</p>
      </aside>
    </div>
    <p class="rr-privacy tiny faint">Frames stay in this tab, are not recorded or uploaded, and are cleared after each request. Camera tracks stop when you pause, stop, or leave this tab.</p>
  </section>
</div>`;

function frameQuestions(includeExpressions) {
  return includeExpressions ? { ...BASE_QUESTIONS, ...EXPRESSION_QUESTIONS } : BASE_QUESTIONS;
}

function scoreMarkup(key, answers) {
  const probability = answers?.[key]?.noul;
  if (typeof probability !== "number" || !Number.isFinite(probability)) return "";
  const value = Math.max(0, Math.min(1, probability));
  const percent = Math.round(value * 100);
  const label = LABELS[key];
  return `<div class="rr-score">
    <div class="rr-score-label"><span>${esc(label)}</span><b>${percent}%</b></div>
    <div class="rr-meter" role="meter" aria-label="${esc(label)} model score, uncalibrated" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}"><i style="width:${percent}%"></i></div>
  </div>`;
}

export function mount(el, { session }) {
  css(new URL("./reaction.css", import.meta.url).href);
  el.innerHTML = TEMPLATE;
  const $ = (name) => el.querySelector(`[data-f="${name}"]`);
  const video = $("video");
  const canvas = document.createElement("canvas");
  canvas.width = FRAME_WIDTH;
  canvas.height = FRAME_HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const gate = $("gate");
  const app = $("app");
  const startButton = el.querySelector('[data-act="start"]');
  const pauseButton = el.querySelector('[data-act="pause"]');
  const stopButton = el.querySelector('[data-act="stop"]');
  const cancelButton = el.querySelector('[data-act="cancel"]');

  let visible = false;
  let starting = false;
  let cameraActive = false;
  let pausedByUser = false;
  let statusMessage = "";
  let startAttempt = 0;
  let stream = null;
  let requestModel = null;
  let previousModel = session.kevala;

  function imageReady(s = session) {
    return !!s.ready && supportsImage(s.info);
  }

  function captureFrame() {
    if (!cameraActive || !stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !context) return null;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!sourceWidth || !sourceHeight) return null;
    const targetAspect = FRAME_WIDTH / FRAME_HEIGHT;
    const sourceAspect = sourceWidth / sourceHeight;
    let cropX = 0;
    let cropY = 0;
    let cropWidth = sourceWidth;
    let cropHeight = sourceHeight;
    if (sourceAspect > targetAspect) {
      cropWidth = sourceHeight * targetAspect;
      cropX = (sourceWidth - cropWidth) / 2;
    } else {
      cropHeight = sourceWidth / targetAspect;
      cropY = (sourceHeight - cropHeight) / 2;
    }
    context.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    try {
      return context.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    } finally {
      context.clearRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    }
  }

  function releaseFrame(frame) {
    // ImageData has no close() method; wipe its pixel buffer before allowing it to be collected.
    frame?.data?.fill?.(0);
    frame?.close?.();
  }

  function updateControls(s = session) {
    const canRun = imageReady(s);
    const cameraAvailable = !!navigator.mediaDevices?.getUserMedia;
    gate.hidden = canRun;
    app.hidden = !canRun;
    startButton.hidden = cameraActive || starting;
    startButton.disabled = !canRun || !visible;
    startButton.textContent = pausedByUser ? "Resume camera" : "Start camera";
    pauseButton.hidden = !cameraActive;
    stopButton.hidden = !cameraActive;
    cancelButton.hidden = !starting;
    $("camera-status").textContent = starting ? "Requesting camera access…" : cameraActive ? (statusMessage ? "Frame analysis issue" : "Camera on") : pausedByUser ? "Camera paused" : statusMessage ? "Camera stopped" : "Camera is off";
    $("camera-detail").textContent = statusMessage || (cameraActive
      ? "Frames are processed locally; Stop camera to end access."
      : pausedByUser
        ? "Camera tracks are stopped. Resume asks for access again."
        : cameraAvailable
          ? "The camera starts only after you press the button."
          : "Camera access is unavailable in this browser. Use a secure page with camera support.");
    $("status-dot").classList.toggle("is-live", cameraActive);
    $("status-dot").classList.toggle("is-pending", starting);
    $("model-label").textContent = canRun ? s.nameOf() : "";
    $("placeholder").hidden = cameraActive;
  }

  function setGate(s) {
    if (s.ready && !supportsImage(s.info)) {
      $("gate-title").textContent = `${s.nameOf()} has no image input`;
      $("gate-detail").textContent = "Choose a model whose loaded metadata advertises image input. For the first demo, select Gemma 4 E2B with its image-capable option in the model picker, then press Load. This view will not load a model or start the camera automatically.";
    } else if (s.status === "loading") {
      $("gate-title").textContent = `Loading ${s.nameOf()}`;
      $("gate-detail").textContent = s.progress?.label || "Wait for the selected model to finish loading, then return here.";
    } else if (s.status === "error") {
      if (s.errorKind === "runtime") {
        $("gate-title").textContent = "Model unavailable";
        $("gate-detail").textContent = `${s.error || "A WebGPU runtime error occurred."} The loaded model was unloaded. Free GPU memory and press Load to retry, or choose another model.`;
      } else {
        $("gate-title").textContent = `Could not load ${s.nameOf()}`;
        $("gate-detail").textContent = `${s.error || "The model did not load."} Select an image-capable model in the picker, then load it before starting the camera.`;
      }
    } else {
      $("gate-title").textContent = "Choose an image-capable model";
      $("gate-detail").textContent = "Select an image-capable model in the model picker and press Load. For the first demo, choose Gemma 4 E2B with its image-capable option. Camera access is requested only when you press Start camera.";
    }
  }

  function paint(response, { latencyMs, fps }) {
    const answers = response?.answers || {};
    const keys = $("extra-questions").checked
      ? Object.keys(LABELS)
      : ["person_present", "smiling"];
    $("scores").innerHTML = keys.map((key) => scoreMarkup(key, answers)).join("") || '<p class="muted small">No scores were returned for this frame.</p>';
    $("latency").textContent = `${Math.round(latencyMs)} ms`;
    $("fps").textContent = fps > 0 ? fps.toFixed(1) : "—";
  }

  function showError(error) {
    statusMessage = error?.message || String(error);
    updateControls();
  }

  const scheduler = createLatestFrameScheduler({
    capture: captureFrame,
    infer: (image) => {
      const model = session.kevala;
      if (!imageReady() || !model) throw new Error("Load an image-capable model before analyzing frames.");
      requestModel = model;
      return model.decide(STATE, frameQuestions($("extra-questions").checked), {
        parts: [{ type: "image", data: image }],
      });
    },
    onResult: (response, metrics) => {
      if (!visible || !cameraActive || requestModel !== session.kevala || !imageReady()) return;
      statusMessage = "";
      updateControls();
      $("camera-detail").textContent = "Only the latest available frame is analyzed; older frames are skipped.";
      paint(response, metrics);
    },
    onError: (error) => {
      if (!visible || !cameraActive || requestModel !== session.kevala) return;
      const message = error?.message || String(error);
      stopCamera(`Frame analysis stopped: ${message}`, { clearScores: true });
      if (error?.code === "WEBGPU_INIT") session.failRuntime(error, requestModel);
    },
    release: releaseFrame,
    intervalMs: MIN_FRAME_INTERVAL_MS,
  });

  function stopTracks() {
    const activeStream = stream;
    stream = null;
    if (!activeStream) return;
    for (const track of activeStream.getTracks()) track.stop();
  }

  function stopCamera(message, { paused = false, clearScores = false } = {}) {
    startAttempt += 1;
    starting = false;
    cameraActive = false;
    pausedByUser = paused;
    statusMessage = message || "";
    scheduler.pause();
    stopTracks();
    video.pause();
    video.srcObject = null;
    context?.clearRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    if (clearScores) {
      $("scores").innerHTML = '<p class="muted small">Start the camera to see model scores.</p>';
      $("fps").textContent = "—";
      $("latency").textContent = "—";
    }
    updateControls();
  }

  async function startCamera() {
    if (starting || cameraActive || !visible || !imageReady()) return;
    const model = session.kevala;
    if (!navigator.mediaDevices?.getUserMedia) {
      showError(new Error("This browser does not provide camera access. Use a secure page with getUserMedia support."));
      return;
    }
    const attempt = ++startAttempt;
    starting = true;
    statusMessage = "";
    updateControls();
    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: FRAME_WIDTH },
          height: { ideal: FRAME_HEIGHT },
          facingMode: "user",
        },
      });
      if (attempt !== startAttempt || !visible || session.kevala !== model || !imageReady()) {
        for (const track of nextStream.getTracks()) track.stop();
        return;
      }
      stream = nextStream;
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (stream === nextStream) stopCamera("The camera track ended.");
        }, { once: true });
      }
      video.srcObject = stream;
      await video.play();
      if (attempt !== startAttempt || !visible || session.kevala !== model || !imageReady()) {
        for (const track of nextStream.getTracks()) track.stop();
        if (stream === nextStream) stream = null;
        if (video.srcObject === nextStream) {
          video.pause();
          video.srcObject = null;
        }
        return;
      }
      pausedByUser = false;
      cameraActive = true;
      $("scores").innerHTML = '<p class="muted small">Waiting for the first frame…</p>';
      $("fps").textContent = "—";
      $("latency").textContent = "—";
      scheduler.start();
    } catch (error) {
      if (attempt === startAttempt) {
        stopTracks();
        video.pause();
        video.srcObject = null;
        showError(error);
      }
    } finally {
      if (attempt === startAttempt) {
        starting = false;
        updateControls();
      }
    }
  }

  function sync(s) {
    if (s.kevala !== previousModel) {
      if (cameraActive || starting) stopCamera("The loaded model changed, so the camera was stopped.");
      previousModel = s.kevala;
    }
    setGate(s);
    updateControls(s);
  }

  function onVisibilityChange() {
    if (document.hidden && (cameraActive || starting)) stopCamera("Camera stopped because this page is hidden.");
  }
  document.addEventListener("visibilitychange", onVisibilityChange);

  el.addEventListener("click", (event) => {
    const action = event.target.closest("[data-act]")?.dataset.act;
    if (action === "open-picker") document.dispatchEvent(new CustomEvent("kevala:open-model-picker"));
    if (action === "start") void startCamera();
    if (action === "cancel") stopCamera("Camera request cancelled.");
    if (action === "pause") stopCamera("Camera paused. Tracks are stopped; press Resume camera to request access again.", { paused: true });
    if (action === "stop") stopCamera("Camera stopped.", { clearScores: true });
  });

  const unsubscribe = session.on(sync);
  sync(session);

  return {
    show() {
      visible = true;
      sync(session);
    },
    hide() {
      visible = false;
      if (cameraActive || starting) stopCamera("Camera stopped because you left this tab.");
    },
    dispose() {
      visible = false;
      unsubscribe();
      document.removeEventListener("visibilitychange", onVisibilityChange);
      scheduler.stop();
      stopCamera("Camera stopped because this view was closed.");
    },
  };
}
