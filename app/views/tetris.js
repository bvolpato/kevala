// Tetris: rendering, input (keyboard with DAS/ARR, touch), the simulation loop and the model
// panel. Game rules live in ../tetris/engine.js, the auto player in ../tetris/ai.js.

import { Game, W, H, HIDDEN, VISIBLE, SOFT_DROP_MS, SHAPES, pieceCells } from "../tetris/engine.js";
import { AutoPlayer, QUESTION, SPEEDS, KEYS, GLYPH, KEY_NAME } from "../tetris/ai.js";
import { css, esc, fmtMs, highlightJSON, logo, modelGate } from "../ui.js";
import { params } from "../session.js";

const COLORS = { I: "#45e0c0", O: "#f6c453", T: "#b48cff", S: "#8be36b", Z: "#ff7a8a", J: "#7aa2ff", L: "#ff9f5a" };
const TYPE_OF = ["", "I", "O", "T", "S", "Z", "J", "L"];
const DAS = 160; // ms before a held key repeats
const ARR = 35; // ms between repeats
const STEP = 1000 / 120; // fixed simulation step
const BEST_KEY = "tetris.best.v1";
const LINE_CLEARS = ["", "SINGLE", "DOUBLE", "TRIPLE", "TETRIS"];
/** `?clip=1`: a full-window 16:9 stage without the site around it, for recording the model play. */
const CLIP = params.get("clip") === "1";

const SPEED_BUTTONS = Object.entries(SPEEDS)
  .map(([id, speed]) => {
    return `<button type="button" class="seg-b" data-speed="${id}" aria-pressed="${id === "normal"}">${speed.label}</button>`;
  })
  .join("");

// laid out by the CSS grid areas: rotate on top, then left, drop and right
const KEY_TILES = ["rotate", "left", "drop", "right"]
  .map((key) => {
    const parts = [
      `<span class="kg">${GLYPH[key]}</span>`,
      `<span class="kn">${KEY_NAME[key]}</span>`,
      `<span class="kv">–</span>`,
      `<span class="kb"><i></i></span>`,
    ];
    return `<div class="key none" data-k="${key}" style="--a:${key}">${parts.join("")}</div>`;
  })
  .join("");

const HTML = `<div class="wrap">
  <div class="t-head">
    <div>
      <h1>Model-guided Tetris</h1>
      <p>Game rules shortlist promising landings the auto controller can reach; the selected model scores their board states.</p>
    </div>
    <div class="t-tools">
      <button type="button" class="btn auto-btn" aria-pressed="false"><span class="led"></span>Auto <kbd>A</kbd></button>
      <div class="seg" role="group" aria-label="Key speed">${SPEED_BUTTONS}</div>
      <button type="button" class="btn reset-btn" title="A new game with new pieces">Reset <kbd>R</kbd></button>
    </div>
  </div>
  <div data-gate></div>
  <div class="game">
    <aside class="side left">
      <div class="box hold-box">
        <div class="box-t">Hold <kbd>C</kbd></div>
        <canvas class="hold" width="120" height="80" aria-label="Held piece"></canvas>
      </div>
      <div class="box stats">
        <div class="stat"><span>Score</span><b data-s="score">0</b></div>
        <div class="stat"><span>Lines</span><b data-s="lines">0</b></div>
        <div class="stat"><span>Level</span><b data-s="level">1</b></div>
        <div class="stat best"><span>Best</span><b data-s="best">0</b></div>
      </div>
    </aside>
    <div class="board-wrap">
      <canvas class="board" aria-label="Game board" tabindex="0"></canvas>
      <div class="overlay">
        <div class="ov-card">
          <h2 class="ov-title">Tetris</h2>
          <p class="ov-text muted">Play yourself, or watch model-guided play.</p>
          <div class="row">
            <button class="btn primary ov-auto" type="button">Watch Auto play</button>
            <button class="btn ov-play" type="button">Play <kbd>Enter</kbd></button>
          </div>
          <p class="tiny faint">Arrows move · Up/X turn · Z turn back · Space drop · C hold · P pause</p>
        </div>
      </div>
      <div class="float"></div>
    </div>
    <aside class="side right">
      <div class="box">
        <div class="box-t">Next</div>
        <canvas class="next" width="120" height="330" aria-label="Next pieces"></canvas>
      </div>
    </aside>
    <section class="ai" aria-label="What the model decides">
      <div class="clip-brand">
        <div class="clip-mark">${logo("tg")}<span>kevala</span></div>
        <h2>Watch model-guided Tetris</h2>
        <p>The selected model on WebGPU or WebAssembly in the browser</p>
      </div>
      <div class="ai-metrics">
        <div title="Full wall time for one model decision, from sending the scoring request until the scores return."><b data-m="ms">–</b><span data-m="where">median model wait</span></div>
        <div title="Landing placements found by the auto controller's rotation, slide, and drop search."><b data-m="spots">–</b><span>landings found</span></div>
        <div title="Distinct board states from the shortlist that were sent to the model."><b data-m="states">–</b><span>states scored</span></div>
      </div>
      <div class="keys">${KEY_TILES}</div>
      <div class="plan" aria-label="Keys for this piece"></div>
      <div class="ai-sub"><span>Best spots</span><span>P(clean stack)</span></div>
      <ol class="spots"><li class="empty">Turn on Auto to watch the model score shortlisted states.</li></ol>
      <details class="seen">
        <summary>What the model read</summary>
        <div class="code"><pre class="json" data-seen></pre></div>
      </details>
      <div class="clip-foot">bvolpato.github.io/kevala</div>
    </section>
  </div>
  <div class="touch" aria-label="Touch controls">
    <button type="button" data-t="hold">Hold</button>
    <button type="button" data-t="ccw" aria-label="Turn counter-clockwise">⟲</button>
    <button type="button" data-t="cw" aria-label="Turn clockwise">⟳</button>
    <button type="button" data-t="pause" aria-label="Pause">❚❚</button>
    <button type="button" data-t="left" aria-label="Move left">◀</button>
    <button type="button" data-t="soft" aria-label="Soft drop">▼</button>
    <button type="button" data-t="right" aria-label="Move right">▶</button>
    <button type="button" data-t="hard" aria-label="Hard drop">⤓</button>
  </div>
  <section class="explain grid-3">
    <div><h3>1 · List</h3><p>The auto controller searches reachable turn, slide, and drop paths using the game's collision rules.</p></div>
    <div><h3>2 · Shortlist</h3><p>Simple game rules choose promising landings and describe the resulting stacks: holes, cleared rows, height and roughness.</p></div>
    <div><h3>3 · Score</h3><p>The selected model scores the shortlist. With multiple states, the piece follows the highest score. These decision models were not trained to play Tetris.</p></div>
  </section>
</div>`;

const TOAST_STYLE = [
  "position:fixed",
  "left:50%",
  "bottom:22px",
  "transform:translateX(-50%)",
  "z-index:60",
  "background:#141925",
  "border:1px solid #2c3749",
  "color:#e7ecf3",
  "padding:9px 14px",
  "border-radius:10px",
  "font-size:.86rem",
  "box-shadow:0 10px 30px rgba(0,0,0,.5)",
  "transition:opacity .3s",
].join(";");

/** "DOUBLE", "T-SPIN SINGLE", ...; a T-spin clears three lines at most. */
function clearName(lines, spin) {
  if (!spin) return LINE_CLEARS[lines];
  return `T-SPIN ${lines < 4 ? LINE_CLEARS[lines] : ""}`.trim();
}

const pct = (p) => `${(p * 100).toFixed(p >= 0.995 ? 0 : 1)}%`;

/** A color `amt` of the way toward white (positive) or black (negative). */
function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const mix = (v) => Math.max(0, Math.min(255, Math.round(amt > 0 ? v + (255 - v) * amt : v * (1 + amt))));
  return `#${[(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => mix(v).toString(16).padStart(2, "0")).join("")}`;
}

export function mount(el, { session }) {
  css(new URL("./tetris.css", import.meta.url).href);
  el.innerHTML = HTML;
  const $ = (s) => el.querySelector(s);
  const ui = {
    board: $(".board"),
    hold: $(".hold"),
    next: $(".next"),
    wrap: $(".board-wrap"),
    overlay: $(".overlay"),
    ovTitle: $(".ov-title"),
    ovText: $(".ov-text"),
    ovPlay: $(".ov-play"),
    ovAuto: $(".ov-auto"),
    float: $(".float"),
    auto: $(".auto-btn"),
    spots: $(".spots"),
    plan: $(".plan"),
    seen: $("[data-seen]"),
  };
  const unGate = modelGate($("[data-gate]"), "watch it play");

  // State

  // every game draws new pieces; `?seed=N` replays one order (for tests and recordings)
  const fixedSeed = Number(params.get("seed")) || 0;
  const newSeed = () => fixedSeed || crypto.getRandomValues(new Uint32Array(1))[0];
  const game = new Game(newSeed());
  game.paused = true;
  let started = false;
  let visible = false;
  let resumeOnShow = false;
  let autoPieces = 0;
  let humanPieces = 0;
  let restartTimer = 0;
  let flashes = [];
  let userTookOver = false;
  let wantAuto = false;
  const model = () => (session.ready ? session.kevala : null);

  const ai = new AutoPlayer(game, {
    getModel: model,
    onDecision: showDecision,
    onPress: (key) => {
      const tile = el.querySelector(`.key[data-k="${key}"]`);
      tile?.classList.add("hit");
      setTimeout(() => tile?.classList.remove("hit"), 90);
    },
    onStatus: showStatus,
    onError: (e) => toast(`Model error: ${e.message}`),
  });
  if (SPEEDS[params.get("speed")]) ai.speed = params.get("speed");

  game.on("spawn", () => ai.onSpawn());
  game.on("lock", ({ rows, points, spin }) => {
    if (ai.enabled) autoPieces++;
    else humanPieces++;
    if (rows.length) flashes.push({ rows: rows.map((y) => y - HIDDEN), t: performance.now() });
    if (rows.length === 4 || (spin && rows.length)) {
      // restart the shake animation
      ui.wrap.classList.remove("shake");
      void ui.wrap.offsetWidth;
      ui.wrap.classList.add("shake");
    }
    if (points) floatText(clearName(rows.length, spin), `+${points}${game.combo > 0 ? ` · combo ${game.combo}` : ""}`);
  });
  game.on("gameover", () => {
    saveBest();
    const summary = `${game.score.toLocaleString()} points · ${game.lines} lines · ${game.pieces} pieces`;
    showOverlay("Game over", `${summary}${ai.enabled ? " · the model was playing" : ""}.`, "Play again");
    if (ai.enabled) {
      ui.ovText.textContent += " A new game starts in 4 s…";
      restartTimer = 4000;
    }
  });

  // Best score

  let best = { score: 0, lines: 0 };
  try {
    best = JSON.parse(localStorage.getItem(BEST_KEY)) || best;
  } catch {}
  function saveBest() {
    if (game.score <= best.score) return;
    best = { score: game.score, lines: game.lines, by: autoPieces > humanPieces ? "model" : "you", at: Date.now() };
    try {
      localStorage.setItem(BEST_KEY, JSON.stringify(best));
    } catch {}
  }

  // Game flow

  function start(auto) {
    ai.reset(); // before the new board exists, so its first spawn is asked about with the new epoch
    game.reset(newSeed());
    autoPieces = humanPieces = 0;
    restartTimer = 0;
    flashes = [];
    started = true;
    game.paused = false;
    hideOverlay();
    ai.setEnabled(!!auto && !!model());
    if (auto) ai.request();
    if (visible) ui.board.focus({ preventScroll: true });
  }

  const PAUSE_TEXT = {
    user: "The game is paused.",
    hidden: "Paused while the tab was hidden.",
    view: "Paused while you were on another page.",
  };

  function togglePause(reason = "user") {
    if (!started || game.over) return;
    if (game.paused) {
      game.paused = false;
      hideOverlay();
      if (ai.enabled && !ai.plan) ai.request();
    } else {
      game.paused = true;
      game.softDrop = false;
      showOverlay("Paused", PAUSE_TEXT[reason] ?? PAUSE_TEXT.user, "Resume");
    }
  }

  function setAuto(on) {
    if (on && !model()) {
      toast(session.status === "loading" ? "The model is still loading; Auto starts when it is ready." : "Loading the model first…");
      wantAuto = true;
      session.load();
      return;
    }
    userTookOver = !on;
    if (on && (!started || game.over)) return start(true);
    if (on && game.paused) togglePause();
    ai.setEnabled(on);
    showStatus();
  }

  function showOverlay(title, text, playLabel) {
    ui.ovTitle.textContent = title;
    ui.ovText.textContent = text;
    ui.ovPlay.innerHTML = `${esc(playLabel)} <kbd>${title === "Paused" ? "P" : "Enter"}</kbd>`;
    ui.ovAuto.classList.toggle("hidden", title === "Paused");
    ui.overlay.classList.remove("hidden");
  }
  function hideOverlay() {
    ui.overlay.classList.add("hidden");
  }

  ui.ovPlay.addEventListener("click", () => {
    if (started && game.paused && !game.over) togglePause();
    else {
      userTookOver = true;
      start(false);
    }
  });
  ui.ovAuto.addEventListener("click", () => setAuto(true));
  ui.auto.addEventListener("click", () => setAuto(!ai.enabled));
  for (const button of el.querySelectorAll("[data-speed]")) {
    button.addEventListener("click", () => {
      ai.speed = button.dataset.speed;
      for (const b of el.querySelectorAll("[data-speed]")) b.setAttribute("aria-pressed", String(b === button));
    });
    button.setAttribute("aria-pressed", String(button.dataset.speed === ai.speed));
  }
  el.querySelector(".reset-btn").addEventListener("click", () => press("reset"));

  // start watching as soon as the model is there, unless the player has taken the controls
  function maybeAutoStart() {
    if (!visible || !model()) return;
    if (wantAuto || (!started && !userTookOver)) {
      wantAuto = false;
      if (started && !game.over) {
        if (game.paused) togglePause();
        ai.setEnabled(true);
      } else start(true);
    }
  }
  const unSession = session.on(() => {
    if (!model() && ai.enabled) ai.setEnabled(false);
    showStatus();
    maybeAutoStart();
  });

  // Input: keyboard with DAS/ARR, touch buttons

  const ACTIONS = {
    ArrowLeft: "left",
    ArrowRight: "right",
    ArrowDown: "soft",
    ArrowUp: "cw",
    KeyX: "cw",
    KeyZ: "ccw",
    Space: "hard",
    KeyC: "hold",
    ShiftLeft: "hold",
    ShiftRight: "hold",
    KeyP: "pause",
    Escape: "pause",
    Enter: "restart",
    KeyR: "reset",
    KeyA: "auto",
  };
  const GAMEPLAY = new Set(["left", "right", "soft", "cw", "ccw", "hard", "hold"]);
  const held = new Map();
  let shiftDir = null;

  function press(action) {
    if (action === "pause") return togglePause();
    if (action === "restart") return started && !game.over ? null : start(ai.enabled);
    if (action === "reset") return start(ai.enabled || (!userTookOver && !!model()));
    if (action === "auto") return setAuto(!ai.enabled);
    if (!GAMEPLAY.has(action)) return;
    if (!started || game.over) {
      if (game.over) return;
      userTookOver = true;
      start(false);
    }
    if (game.paused) return;
    if (ai.takeover()) {
      userTookOver = true;
      toast("You have the controls. Press A to hand back to the model.");
    }
    switch (action) {
      case "left":
      case "right":
        held.set(action, { t: 0, last: 0 });
        shiftDir = action;
        game.move(action === "left" ? -1 : 1);
        break;
      case "soft":
        held.set("soft", { t: 0 });
        game.softDrop = true;
        break;
      case "cw":
        game.rotate(1);
        break;
      case "ccw":
        game.rotate(-1);
        break;
      case "hard":
        game.hardDrop();
        break;
      case "hold":
        game.holdPiece();
        break;
    }
  }

  function release(action) {
    held.delete(action);
    if (action === "soft" && !ai.enabled) game.softDrop = false;
    if (action === shiftDir) shiftDir = held.has("left") ? "left" : held.has("right") ? "right" : null;
  }

  function repeatInput(dt) {
    if (!shiftDir || !game.active) return;
    const hold = held.get(shiftDir);
    if (!hold) return;
    hold.t += dt;
    if (hold.t < DAS) return;
    hold.last += dt;
    while (hold.last >= ARR) {
      hold.last -= ARR;
      if (!game.move(shiftDir === "left" ? -1 : 1)) {
        hold.last = 0;
        break;
      }
    }
  }

  let keyboard = true;
  const onKeyDown = (e) => {
    if (!visible || !keyboard) return;
    if (e.target.closest?.("input, textarea, select, .mpanel")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const action = ACTIONS[e.code];
    if (!action) return;
    // Space and Enter keep their usual meaning on focused buttons and links outside the game
    if ((e.code === "Space" || e.code === "Enter") && e.target.closest?.("button, a, summary") && !e.target.closest(".game, .t-tools")) return;
    e.preventDefault();
    if (e.repeat) return;
    press(action);
  };
  const onKeyUp = (e) => {
    const action = ACTIONS[e.code];
    if (action) release(action);
  };
  const onBlur = () => {
    for (const action of [...held.keys()]) release(action);
  };
  addEventListener("keydown", onKeyDown);
  addEventListener("keyup", onKeyUp);
  addEventListener("blur", onBlur);

  // each touch button's data-t is the action it presses
  for (const button of el.querySelectorAll(".touch [data-t]")) {
    const action = button.dataset.t;
    button.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      button.classList.add("down");
      button.setPointerCapture?.(e.pointerId);
      press(action);
    });
    const up = () => {
      button.classList.remove("down");
      release(action);
    };
    button.addEventListener("pointerup", up);
    button.addEventListener("pointercancel", up);
    button.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && visible && started && !game.paused && !game.over && !ai.enabled) togglePause("hidden");
    lastFrame = performance.now();
  });

  // Rendering

  let cell = 28;
  let dpr = 1;
  const sprites = new Map();

  function layout() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const narrow = innerWidth <= 640;
    // the page keeps room for the header and title; the clip stage gives the board the full height
    const byHeight = Math.floor((innerHeight - (CLIP ? 96 : 200)) / VISIBLE);
    const byWidth = narrow ? Math.floor((innerWidth - 48) / W) : 64;
    cell = Math.max(16, Math.min(CLIP ? 64 : 38, byHeight, byWidth));
    el.querySelector(".game").style.setProperty("--cell", `${cell}px`);
    ui.board.width = W * cell * dpr;
    ui.board.height = VISIBLE * cell * dpr;
    sprites.clear();
    const sideCanvases = [
      [ui.next, narrow ? 240 : 120, narrow ? 70 : 330],
      [ui.hold, 120, narrow ? 70 : 80],
    ];
    for (const [canvas, w, h] of sideCanvases) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.aspectRatio = `${w} / ${h}`;
    }
  }
  addEventListener("resize", () => visible && layout());
  layout();

  /** One block of `color`, `size` CSS pixels wide, drawn once and reused. */
  function sprite(color, size) {
    const key = `${color}|${size}`;
    if (sprites.has(key)) return sprites.get(key);
    const px = Math.round(size * dpr);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = px;
    const ctx = canvas.getContext("2d");
    const radius = px * 0.2;
    const inset = Math.max(1, px * 0.05);
    const outline = () => {
      ctx.beginPath();
      ctx.roundRect(inset, inset, px - 2 * inset, px - 2 * inset, radius);
    };
    outline();
    const grad = ctx.createLinearGradient(0, 0, px, px);
    grad.addColorStop(0, shade(color, 0.22));
    grad.addColorStop(0.55, color);
    grad.addColorStop(1, shade(color, -0.28));
    ctx.fillStyle = grad;
    ctx.fill();
    // a light band on top and a dark one at the bottom
    ctx.save();
    ctx.clip();
    ctx.fillStyle = "rgba(255,255,255,0.22)";
    ctx.fillRect(inset, inset, px, px * 0.12);
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    ctx.fillRect(inset, px - inset - px * 0.1, px, px * 0.1);
    ctx.restore();
    ctx.lineWidth = Math.max(1, px * 0.04);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    outline();
    ctx.stroke();
    sprites.set(key, canvas);
    return canvas;
  }

  function drawBoard(now) {
    const ctx = ui.board.getContext("2d");
    const px = cell * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ui.board.width, ui.board.height);
    ctx.fillStyle = "rgba(255,255,255,0.035)";
    for (let y = 0; y < VISIBLE; y++) {
      for (let x = 0; x < W; x++) ctx.fillRect(x * px + px / 2 - dpr, y * px + px / 2 - dpr, 2 * dpr, 2 * dpr);
    }
    ctx.strokeStyle = "rgba(255,255,255,0.025)";
    ctx.lineWidth = 1;
    for (let x = 1; x < W; x++) {
      ctx.beginPath();
      ctx.moveTo(x * px + 0.5, 0);
      ctx.lineTo(x * px + 0.5, VISIBLE * px);
      ctx.stroke();
    }
    for (let y = HIDDEN; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = game.board[y * W + x];
        if (!v) continue;
        ctx.globalAlpha = game.over ? 0.35 : 1;
        ctx.drawImage(sprite(COLORS[TYPE_OF[v]], cell), x * px, (y - HIDDEN) * px);
      }
    }
    ctx.globalAlpha = 1;
    if (game.piece && !game.over) {
      // no landing shadow and no target outline: where the piece lands stays a surprise
      const piece = game.piece;
      // Render the fall accumulator between rows. This changes only the picture; collision,
      // gravity, lock timing, and input continue to use the game's fixed-step integer state.
      const autoDrop = ai.enabled && ai.plan?.pieceId === game.pieceId && ai.plan.dropping;
      const rowMs = game.softDrop ? Math.min(game.gravityMs, SOFT_DROP_MS) : game.gravityMs;
      const fall = game.active && !game.onGround
        ? Math.min(1, autoDrop ? ai.timer / SPEEDS[ai.speed].dropGap : game.fall / rowMs)
        : 0;
      ctx.globalAlpha = game.onGround ? 1 - 0.35 * Math.min(1, game.lockTimer / 500) : 1;
      for (const [x, y] of pieceCells(piece)) {
        if (y >= HIDDEN) ctx.drawImage(sprite(COLORS[piece.type], cell), x * px, (y - HIDDEN + fall) * px);
      }
      ctx.globalAlpha = 1;
    }
    flashes = flashes.filter((f) => now - f.t < 380);
    for (const flash of flashes) {
      const k = (now - flash.t) / 380;
      ctx.fillStyle = `rgba(230,245,255,${0.75 * (1 - k)})`;
      for (const row of flash.rows) {
        const grow = k * px * 0.5;
        ctx.fillRect(0, row * px - grow / 2, W * px, px + grow);
      }
    }
  }

  /** Draws a piece of `type` centered on (cx, cy), in blocks `size` CSS pixels wide. */
  function drawPiece(ctx, type, cx, cy, size, dim = false) {
    const cells = SHAPES[type][0];
    const xs = cells.map(([x]) => x);
    const ys = cells.map(([, y]) => y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const w = Math.max(...xs) - minX + 1;
    const h = Math.max(...ys) - minY + 1;
    const px = size * dpr;
    ctx.globalAlpha = dim ? 0.35 : 1;
    for (const [x, y] of cells) {
      ctx.drawImage(sprite(COLORS[type], size), cx - (w * px) / 2 + (x - minX) * px, cy - (h * px) / 2 + (y - minY) * px);
    }
    ctx.globalAlpha = 1;
  }

  function drawSide() {
    const holdCtx = ui.hold.getContext("2d");
    holdCtx.clearRect(0, 0, ui.hold.width, ui.hold.height);
    if (game.hold) drawPiece(holdCtx, game.hold, ui.hold.width / 2, ui.hold.height / 2, 22, game.holdUsed);
    const nextCtx = ui.next.getContext("2d");
    nextCtx.clearRect(0, 0, ui.next.width, ui.next.height);
    // on narrow screens the queue is a row of three, otherwise a column of five
    const row = ui.next.width > ui.next.height;
    const count = row ? 3 : 5;
    for (let i = 0; i < count; i++) {
      const type = game.queue[i];
      if (!type) continue;
      if (row) drawPiece(nextCtx, type, ((i + 0.5) * ui.next.width) / count, ui.next.height / 2, i === 0 ? 18 : 15);
      else drawPiece(nextCtx, type, ui.next.width / 2, (i === 0 ? 34 : 34 + 70 * i) * dpr, i === 0 ? 22 : 18);
    }
  }

  const statEls = {};
  for (const key of ["score", "lines", "level", "best"]) {
    statEls[key] = el.querySelector(`[data-s="${key}"]`);
  }
  const shownStats = {};
  function setStat(key, value) {
    if (shownStats[key] === value) return;
    shownStats[key] = value;
    statEls[key].textContent = value;
  }
  function drawStats() {
    setStat("score", game.score.toLocaleString());
    setStat("lines", String(game.lines));
    setStat("level", String(game.level));
    setStat("best", Math.max(best.score, game.score).toLocaleString());
  }

  function floatText(title, sub) {
    const span = document.createElement("span");
    span.innerHTML = `${esc(title)}<small>${esc(sub)}</small>`;
    ui.float.replaceChildren(span);
  }

  let toastTimer = 0;
  function toast(msg) {
    let box = document.getElementById("toast");
    if (!box) {
      box = document.createElement("div");
      box.id = "toast";
      box.style.cssText = TOAST_STYLE;
      document.body.appendChild(box);
    }
    box.textContent = msg;
    box.style.opacity = "1";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (box.style.opacity = "0"), 2600);
  }

  // Model panel

  function showStatus() {
    ui.auto.setAttribute("aria-pressed", String(ai.enabled));
    ui.auto.classList.toggle("thinking", ai.busy);
    const backend = session.info?.backend;
    const where = backend ? `median wait · ${backend === "webgpu" ? "WebGPU" : "CPU"}` : "median model wait";
    el.querySelector('[data-m="where"]').textContent = where;
    if (!ai.stats.ms.length) el.querySelector('[data-m="ms"]').textContent = ai.busy ? "…" : "–";
  }

  function spotHTML(spot, i) {
    return [
      `<li class="${i === 0 ? "chosen" : ""}">`,
      `<div class="rt"><span>${esc(spot.group[0].summary)}</span><b>${pct(spot.p)}</b></div>`,
      `<div class="pbar"><i style="width:0"></i></div>`,
      `</li>`,
    ].join("");
  }

  const decisionWaits = [];
  function showDecision(decision) {
    decisionWaits.push(decision.ms);
    if (decisionWaits.length > 10) decisionWaits.shift();
    const sortedWaits = [...decisionWaits].sort((a, b) => a - b);
    const middle = sortedWaits.length >> 1;
    const median = sortedWaits.length % 2
      ? sortedWaits[middle]
      : (sortedWaits[middle - 1] + sortedWaits[middle]) / 2;
    const msEl = el.querySelector('[data-m="ms"]');
    msEl.textContent = fmtMs(median);
    msEl.parentElement.title =
      `Full wall wait for this piece's model decision: ${fmtMs(decision.ms)} to score ${decision.states} distinct states ` +
      `(${decision.timing?.tokens ?? "?"} tokens). Displayed value is the median of the last ${decisionWaits.length} decisions.`;
    el.querySelector('[data-m="spots"]').textContent = String(decision.spots);
    el.querySelector('[data-m="states"]').textContent = String(decision.states);
    const top = decision.scored.slice(0, 3);
    ui.spots.innerHTML = top.map(spotHTML).join("");
    requestAnimationFrame(() => {
      ui.spots.querySelectorAll(".pbar i").forEach((bar, i) => (bar.style.width = `${(top[i].p * 100).toFixed(1)}%`));
    });
    ui.seen.innerHTML = highlightJSON(JSON.stringify({ state: decision.best.text, questions: QUESTION }, null, 2));
    showStatus();
  }

  function planHTML(plan) {
    return plan.keys
      .map((key, i) => {
        const state = i < plan.pressed ? "done" : i === plan.pressed ? "now" : "";
        return `<kbd class="${state}" title="${KEY_NAME[key]}">${GLYPH[key]}</kbd>`;
      })
      .join("");
  }

  // the keys and the plan strip follow the piece every frame; the DOM changes only when they do
  let keySig = "";
  function drawKeys() {
    const values = ai.enabled ? ai.values() : null;
    const next = ai.enabled ? ai.next() : null;
    const plan = ai.enabled && ai.plan && ai.plan.pieceId === game.pieceId ? ai.plan : null;
    const sig = `${JSON.stringify(values)}|${next}|${plan ? `${plan.pressed}/${plan.keys.length}` : ai.busy}`;
    if (sig === keySig) return;
    keySig = sig;
    for (const key of KEYS) {
      const tile = el.querySelector(`.key[data-k="${key}"]`);
      const value = values?.[key];
      tile.classList.toggle("none", value == null);
      tile.classList.toggle("next", key === next);
      tile.querySelector(".kv").textContent = value == null ? "–" : pct(value);
      tile.querySelector(".kb i").style.width = `${value == null ? 0 : (value * 100).toFixed(1)}%`;
    }
    if (plan) ui.plan.innerHTML = planHTML(plan);
    else ui.plan.innerHTML = ai.enabled && ai.busy ? `<span class="pl-t">scoring…</span>` : "";
  }

  // The loop: fixed steps, bounded catch-up; stops while the view is hidden

  let lastFrame = performance.now();
  let acc = 0;
  let raf = 0;
  function frame(now) {
    raf = 0;
    if (!visible) return;
    const dt = Math.min(250, now - lastFrame);
    lastFrame = now;
    acc += dt;
    let steps = 0;
    while (acc >= STEP && steps < 32) {
      tick(STEP);
      acc -= STEP;
      steps++;
    }
    if (steps === 32) acc = 0;
    drawBoard(now);
    drawSide();
    drawStats();
    drawKeys();
    raf = requestAnimationFrame(frame);
  }

  function tick(dt) {
    if (restartTimer > 0) {
      restartTimer -= dt;
      if (restartTimer <= 0 && game.over && ai.enabled) start(true);
    }
    if (!started || game.paused || game.over) return;
    repeatInput(dt);
    ai.tick(dt);
    game.step(dt);
  }

  // clicking a game control hands the keyboard straight back to the board
  for (const button of el.querySelectorAll(".game button, .t-tools button")) {
    button.addEventListener("click", () => ui.board.focus({ preventScroll: true }));
  }
  if (params.get("touch") === "1") el.querySelector(".touch").style.display = "grid";

  showStatus();
  // read-only handle for tests and recordings; ignoreKeys() stops stray typing from taking over
  window.tetris = { game, ai, start, setAuto, ignoreKeys: (on = true) => (keyboard = !on) };

  return {
    show() {
      visible = true;
      document.body.classList.toggle("t-clip", CLIP);
      layout();
      lastFrame = performance.now();
      if (!raf) raf = requestAnimationFrame(frame);
      if (resumeOnShow && started && game.paused && !game.over) togglePause();
      resumeOnShow = false;
      maybeAutoStart();
    },
    hide() {
      visible = false;
      document.body.classList.remove("t-clip");
      onBlur();
      if (started && !game.paused && !game.over) {
        // an Auto game picks up where it was; a human game waits for an explicit resume
        resumeOnShow = ai.enabled;
        togglePause("view");
      }
    },
    destroy() {
      unGate();
      unSession();
      removeEventListener("keydown", onKeyDown);
      removeEventListener("keyup", onKeyUp);
      removeEventListener("blur", onBlur);
    },
  };
}
