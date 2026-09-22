// Tetris: rendering, input (keyboard with DAS/ARR, touch), the simulation loop and the model
// panel. Game rules live in ../tetris/engine.js, the auto player in ../tetris/ai.js.

import { Game, W, H, HIDDEN, VISIBLE, SHAPES, pieceCells } from "../tetris/engine.js";
import { AutoPlayer, QUESTION, SPEEDS, KEYS, GLYPH, KEY_NAME } from "../tetris/ai.js";
import { css, esc, fmtMs, backendBadge, highlightJSON, modelGate } from "../ui.js";
import { params } from "../session.js";

const COLORS = { I: "#45e0c0", O: "#f6c453", T: "#b48cff", S: "#8be36b", Z: "#ff7a8a", J: "#7aa2ff", L: "#ff9f5a" };
const TYPE_OF = ["", "I", "O", "T", "S", "Z", "J", "L"];
const DEMO_SEED = 3;
const DAS = 160; // ms before a held key repeats
const ARR = 35; // ms between repeats
const STEP = 1000 / 120; // fixed simulation step
const BEST_KEY = "tetris.best.v1";

const HTML = `<div class="wrap">
  <div class="t-head">
    <div>
      <div class="eyebrow">Demo · real-time control</div>
      <h1>Tetris, played by a decision model</h1>
      <p>Each new piece, the model reads every place it could land and scores them in one pass. Then it presses the keys to get there: turn, left, right, drop. Everything runs in this tab.</p>
    </div>
  </div>
  <div data-gate></div>
  <div class="game">
    <aside class="side left">
      <div class="box">
        <div class="box-t">Hold <kbd>C</kbd></div>
        <canvas class="hold" width="120" height="80" aria-label="Held piece"></canvas>
      </div>
      <div class="box stats">
        <div class="stat"><span>Score</span><b data-s="score">0</b></div>
        <div class="stat"><span>Lines</span><b data-s="lines">0</b></div>
        <div class="stat"><span>Level</span><b data-s="level">1</b></div>
        <div class="stat"><span>Time</span><b data-s="time">0:00</b></div>
        <div class="stat"><span>Pieces</span><b data-s="pieces">0</b></div>
        <div class="stat best"><span>Best</span><b data-s="best">0</b></div>
      </div>
    </aside>
    <div class="board-wrap">
      <canvas class="board" aria-label="Game board" tabindex="0"></canvas>
      <div class="overlay">
        <div class="ov-card">
          <h2 class="ov-title">Tetris</h2>
          <p class="ov-text muted">Play yourself, or let the model play.</p>
          <div class="row">
            <button class="btn primary ov-auto" type="button">Watch the model play</button>
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
      <div class="box mode">
        <div class="box-t">Pieces</div>
        <div class="seg" role="group" aria-label="Piece order">
          <button type="button" class="seg-b" data-seed="demo" aria-pressed="true">Demo</button>
          <button type="button" class="seg-b" data-seed="random" aria-pressed="false">Random</button>
        </div>
        <div class="tiny faint seed-label"></div>
      </div>
    </aside>
    <section class="ai card" aria-label="Model panel">
      <div class="ai-top">
        <button type="button" class="btn auto-btn" aria-pressed="false"><span class="led"></span>Auto <kbd>A</kbd></button>
        <div class="seg" role="group" aria-label="Key speed">
          ${Object.entries(SPEEDS).map(([k, v]) => `<button type="button" class="seg-b" data-speed="${k}" aria-pressed="${k === "normal"}">${v.label}</button>`).join("")}
        </div>
      </div>
      <div class="ai-model"></div>
      <div class="ai-metrics">
        <div title="Time for the model to score every landing spot of this piece"><span>Model</span><b data-m="ms">–</b></div>
        <div title="Landing spots, and the distinct descriptions sent in one pass"><span>Spots</span><b data-m="spots">–</b></div>
        <div><span>Pieces</span><b data-m="dec">0</b></div>
        <div title="Answers that arrived after the piece changed, thrown away"><span>Stale</span><b data-m="drop">0</b></div>
      </div>
      <div class="ai-sub"><span>Keys</span><span class="tiny faint">best model score among the spots each key leads to</span></div>
      <div class="keys">
        ${["rotate", "left", "drop", "right"].map((k) => `<div class="key none" data-k="${k}" style="--a:${k}"><span class="kg">${GLYPH[k]}</span><span class="kn">${KEY_NAME[k]}</span><span class="kv">–</span><span class="kb"><i></i></span></div>`).join("")}
      </div>
      <div class="ai-sub"><span>Plan for this piece</span></div>
      <div class="plan"><span class="pl-t">Waiting for the model…</span></div>
      <div class="ai-sub"><span>Where it is going</span><span class="tiny faint">P(the stack looks clean)</span></div>
      <ol class="spots"><li class="empty faint small">Turn on Auto to see how the model scores every landing spot.</li></ol>
      <details class="seen"><summary>What the model read for the chosen spot</summary><div class="code"><pre class="json" data-seen></pre></div></details>
      <div class="ai-sub"><span>Recent pieces</span></div>
      <ol class="trail"></ol>
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
  <section class="explain">
    <h2 style="font-size:1.4rem">How the model plays</h2>
    <p class="muted" style="max-width:78ch;margin-bottom:18px">When a piece appears, the code lists every place it can land and describes each outcome in plain words. The model gets all of them in one batched pass with a single yes/no question: <em>“Does the stack look clean after this move?”</em>. The piece then heads for the spot with the highest P(yes), one key at a time, while gravity keeps pulling it down; its drop is a fast fall, not a jump. The code measures and describes; the model decides.</p>
    <div class="grid-3">
      <div class="card pad"><h3>1 · List the spots</h3><p class="muted small">Every turn of the piece and every column it can slide to, dropped with the same collision and wall-kick code the game uses. Spots that read the same are merged.</p></div>
      <div class="card pad"><h3>2 · Describe them</h3><p class="muted small">Holes buried, rows completed, and whether the surface and height end up better or worse than with most other spots. Numbers and comparisons are computed in code and stated in words.</p></div>
      <div class="card pad"><h3>3 · Ask once, press keys</h3><p class="muted small">One <code>decideMany</code> pass scores every spot. The keys panel shows, for each key, the best score among the spots it leads toward; the piece presses the top one until it lands.</p></div>
    </div>
    <p class="tiny faint" style="margin-top:14px">Asking the model for one key at a time (left, right, turn or drop, with what each would lead to) was tried first and played far worse: it cleared no lines in 40 pieces. The model's probabilities are its own scores on a task it was not trained for, not a measure of skill.</p>
  </section>
</div>`;

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
    trail: $(".trail"),
    plan: $(".plan"),
    aiModel: $(".ai-model"),
    seen: $("[data-seen]"),
  };
  const unGate = modelGate($("[data-gate]"), "watch it play");

  // -------------------------------------------------------------------------------------------
  // state

  let seedMode = params.get("seed") ? "custom" : "demo";
  const customSeed = Number(params.get("seed")) || DEMO_SEED;
  const newSeed = () => (seedMode === "demo" ? DEMO_SEED : seedMode === "custom" ? customSeed : crypto.getRandomValues(new Uint32Array(1))[0]);
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
  let lastDecision = null;
  let pieceKeys = [];
  let wantAuto = false;
  const model = () => (session.ready ? session.kevala : null);

  const ai = new AutoPlayer(game, {
    getModel: model,
    onDecision: showDecision,
    onPress: (k) => {
      pieceKeys.push(k);
      const b = el.querySelector(`.key[data-k="${k}"]`);
      b?.classList.add("hit");
      setTimeout(() => b?.classList.remove("hit"), 90);
    },
    onStatus: showStatus,
    onError: (e) => toast(`Model error: ${e.message}`),
  });
  if (SPEEDS[params.get("speed")]) ai.speed = params.get("speed");

  game.on("spawn", () => {
    pieceKeys = [];
    ai.onSpawn();
  });
  game.on("lock", ({ rows, points, spin, piece }) => {
    if (ai.enabled) {
      autoPieces++;
      if (lastDecision) addTrail(piece.type, pieceKeys, lastDecision.best.p, lastDecision.ms, rows.length);
    } else humanPieces++;
    if (rows.length) flashes.push({ rows: rows.map((y) => y - HIDDEN), t: performance.now() });
    if (rows.length === 4 || (spin && rows.length)) {
      ui.wrap.classList.remove("shake");
      void ui.wrap.offsetWidth;
      ui.wrap.classList.add("shake");
    }
    if (points) {
      const name = spin ? `T-SPIN ${["", "SINGLE", "DOUBLE", "TRIPLE"][rows.length] || ""}` : ["", "SINGLE", "DOUBLE", "TRIPLE", "TETRIS"][rows.length];
      floatText(name.trim(), `+${points}${game.combo > 0 ? ` · combo ${game.combo}` : ""}`);
    }
  });
  game.on("gameover", () => {
    saveBest();
    showOverlay("Game over", `${game.score.toLocaleString()} points · ${game.lines} lines · ${game.pieces} pieces${ai.enabled ? " · the model was playing" : ""}.`, "Play again");
    if (ai.enabled) {
      ui.ovText.textContent += " A new game starts in 4 s…";
      restartTimer = 4000;
    }
  });

  // -------------------------------------------------------------------------------------------
  // best score

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

  // -------------------------------------------------------------------------------------------
  // game flow

  function start(auto) {
    ai.reset(); // before the new board exists, so its first spawn is asked about with the new epoch
    game.reset(newSeed());
    autoPieces = humanPieces = 0;
    restartTimer = 0;
    flashes = [];
    started = true;
    game.paused = false;
    hideOverlay();
    showSeed();
    ui.trail.innerHTML = "";
    ai.setEnabled(!!auto && !!model());
    if (auto) ai.request();
    if (visible) ui.board.focus({ preventScroll: true });
  }

  function togglePause(reason = "user") {
    if (!started || game.over) return;
    if (game.paused) {
      game.paused = false;
      hideOverlay();
      if (ai.enabled && !ai.plan) ai.request();
    } else {
      game.paused = true;
      game.softDrop = false;
      showOverlay("Paused", reason === "hidden" ? "Paused while the tab was hidden." : reason === "view" ? "Paused while you were on another page." : "Take a breath.", "Resume");
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
  for (const b of el.querySelectorAll("[data-speed]")) {
    b.addEventListener("click", () => {
      ai.speed = b.dataset.speed;
      el.querySelectorAll("[data-speed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    });
    b.setAttribute("aria-pressed", String(b.dataset.speed === ai.speed));
  }
  for (const b of el.querySelectorAll("[data-seed]")) {
    b.addEventListener("click", () => {
      seedMode = b.dataset.seed;
      el.querySelectorAll("[data-seed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      showSeed(true);
    });
  }
  function showSeed(pending) {
    const s = $(".seed-label");
    if (pending && started) s.textContent = seedMode === "demo" ? `Next game: demo order` : "Next game: random";
    else s.textContent = seedMode === "random" ? `Random · seed ${game.seed}` : `Seed ${game.seed}, same pieces every time`;
  }
  showSeed();

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

  // -------------------------------------------------------------------------------------------
  // input: keyboard with DAS/ARR, touch buttons

  const ACTIONS = {
    ArrowLeft: "left", ArrowRight: "right", ArrowDown: "soft", ArrowUp: "cw", KeyX: "cw", KeyZ: "ccw",
    Space: "hard", KeyC: "hold", ShiftLeft: "hold", ShiftRight: "hold", KeyP: "pause", Escape: "pause", Enter: "restart", KeyA: "auto",
  };
  const GAMEPLAY = new Set(["left", "right", "soft", "cw", "ccw", "hard", "hold"]);
  const held = new Map();
  let shiftDir = null;

  function press(action) {
    if (action === "pause") return togglePause();
    if (action === "restart") return started && !game.over ? null : start(ai.enabled);
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
    const h = held.get(shiftDir);
    if (!h) return;
    h.t += dt;
    if (h.t < DAS) return;
    h.last += dt;
    while (h.last >= ARR) {
      h.last -= ARR;
      if (!game.move(shiftDir === "left" ? -1 : 1)) {
        h.last = 0;
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
    if ((e.code === "Space" || e.code === "Enter") && e.target.closest?.("button, a, summary") && !e.target.closest(".game")) return;
    e.preventDefault();
    if (e.repeat) return;
    press(action);
  };
  const onKeyUp = (e) => {
    const action = ACTIONS[e.code];
    if (action) release(action);
  };
  const onBlur = () => {
    for (const a of [...held.keys()]) release(a);
  };
  addEventListener("keydown", onKeyDown);
  addEventListener("keyup", onKeyUp);
  addEventListener("blur", onBlur);

  for (const b of el.querySelectorAll(".touch [data-t]")) {
    const a = { left: "left", right: "right", soft: "soft", cw: "cw", ccw: "ccw", hard: "hard", hold: "hold", pause: "pause" }[b.dataset.t];
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      b.classList.add("down");
      b.setPointerCapture?.(e.pointerId);
      press(a);
    });
    const up = () => {
      b.classList.remove("down");
      release(a);
    };
    b.addEventListener("pointerup", up);
    b.addEventListener("pointercancel", up);
    b.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && visible && started && !game.paused && !game.over && !ai.enabled) togglePause("hidden");
    last = performance.now();
  });

  // -------------------------------------------------------------------------------------------
  // rendering

  let cell = 28;
  let dpr = 1;
  const sprites = new Map();

  function layout() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const narrow = innerWidth <= 640;
    const byH = Math.floor((innerHeight - 250) / VISIBLE);
    const byW = narrow ? Math.floor((innerWidth - 48) / W) : 40;
    cell = Math.max(16, Math.min(32, byH, byW));
    el.querySelector(".game").style.setProperty("--cell", `${cell}px`);
    ui.board.width = W * cell * dpr;
    ui.board.height = VISIBLE * cell * dpr;
    sprites.clear();
    const nw = narrow ? 240 : 120;
    const nh = narrow ? 70 : 330;
    for (const [c, w, h] of [[ui.next, nw, nh], [ui.hold, 120, narrow ? 70 : 80]]) {
      c.width = w * dpr;
      c.height = h * dpr;
      c.style.aspectRatio = `${w} / ${h}`;
    }
  }
  addEventListener("resize", () => visible && layout());
  layout();

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const f = (v) => Math.max(0, Math.min(255, Math.round(amt > 0 ? v + (255 - v) * amt : v * (1 + amt))));
    return `#${[(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => f(v).toString(16).padStart(2, "0")).join("")}`;
  }

  function sprite(color, size) {
    const key = `${color}|${size}`;
    if (sprites.has(key)) return sprites.get(key);
    const s = Math.round(size * dpr);
    const c = document.createElement("canvas");
    c.width = c.height = s;
    const g = c.getContext("2d");
    const r = s * 0.2;
    const inset = Math.max(1, s * 0.05);
    const path = () => {
      g.beginPath();
      g.roundRect(inset, inset, s - 2 * inset, s - 2 * inset, r);
    };
    path();
    const grad = g.createLinearGradient(0, 0, s, s);
    grad.addColorStop(0, shade(color, 0.22));
    grad.addColorStop(0.55, color);
    grad.addColorStop(1, shade(color, -0.28));
    g.fillStyle = grad;
    g.fill();
    g.save();
    g.clip();
    g.fillStyle = "rgba(255,255,255,0.22)";
    g.fillRect(inset, inset, s, s * 0.12);
    g.fillStyle = "rgba(0,0,0,0.18)";
    g.fillRect(inset, s - inset - s * 0.1, s, s * 0.1);
    g.restore();
    g.lineWidth = Math.max(1, s * 0.04);
    g.strokeStyle = "rgba(255,255,255,0.18)";
    path();
    g.stroke();
    sprites.set(key, c);
    return c;
  }

  function drawBoard(now) {
    const g = ui.board.getContext("2d");
    const c = cell * dpr;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, ui.board.width, ui.board.height);
    g.fillStyle = "rgba(255,255,255,0.035)";
    for (let y = 0; y < VISIBLE; y++) for (let x = 0; x < W; x++) g.fillRect(x * c + c / 2 - dpr, y * c + c / 2 - dpr, 2 * dpr, 2 * dpr);
    g.strokeStyle = "rgba(255,255,255,0.025)";
    g.lineWidth = 1;
    for (let x = 1; x < W; x++) {
      g.beginPath();
      g.moveTo(x * c + 0.5, 0);
      g.lineTo(x * c + 0.5, VISIBLE * c);
      g.stroke();
    }
    const dead = game.over;
    for (let y = HIDDEN; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = game.board[y * W + x];
        if (!v) continue;
        g.globalAlpha = dead ? 0.35 : 1;
        g.drawImage(sprite(COLORS[TYPE_OF[v]], cell), x * c, (y - HIDDEN) * c);
      }
    }
    g.globalAlpha = 1;
    if (game.piece && !game.over) {
      // no landing shadow and no target outline: where the piece lands stays a surprise
      const p = game.piece;
      g.globalAlpha = game.onGround ? 1 - 0.35 * Math.min(1, game.lockTimer / 500) : 1;
      for (const [x, y] of pieceCells(p)) if (y >= HIDDEN) g.drawImage(sprite(COLORS[p.type], cell), x * c, (y - HIDDEN) * c);
      g.globalAlpha = 1;
    }
    flashes = flashes.filter((f) => now - f.t < 380);
    for (const f of flashes) {
      const k = (now - f.t) / 380;
      g.fillStyle = `rgba(230,245,255,${0.75 * (1 - k)})`;
      for (const r of f.rows) {
        const grow = k * c * 0.5;
        g.fillRect(0, r * c - grow / 2, W * c, c + grow);
      }
    }
  }

  function drawPiece(g, type, cx, cy, size, dim = false) {
    const cells = SHAPES[type][0];
    const xs = cells.map(([x]) => x);
    const ys = cells.map(([, y]) => y);
    const w = Math.max(...xs) - Math.min(...xs) + 1;
    const h = Math.max(...ys) - Math.min(...ys) + 1;
    const s = size * dpr;
    g.globalAlpha = dim ? 0.35 : 1;
    for (const [x, y] of cells) g.drawImage(sprite(COLORS[type], size), cx - (w * s) / 2 + (x - Math.min(...xs)) * s, cy - (h * s) / 2 + (y - Math.min(...ys)) * s);
    g.globalAlpha = 1;
  }

  function drawSide() {
    const hg = ui.hold.getContext("2d");
    hg.clearRect(0, 0, ui.hold.width, ui.hold.height);
    if (game.hold) drawPiece(hg, game.hold, ui.hold.width / 2, ui.hold.height / 2, 22, game.holdUsed);
    const ng = ui.next.getContext("2d");
    ng.clearRect(0, 0, ui.next.width, ui.next.height);
    const narrow = ui.next.width > ui.next.height;
    const n = narrow ? 3 : 5;
    for (let i = 0; i < n; i++) {
      const t = game.queue[i];
      if (!t) continue;
      if (narrow) drawPiece(ng, t, ((i + 0.5) * ui.next.width) / n, ui.next.height / 2, i === 0 ? 18 : 15);
      else drawPiece(ng, t, ui.next.width / 2, (i === 0 ? 34 : 34 + 70 * i) * dpr, i === 0 ? 22 : 18);
    }
  }

  const statEls = {};
  for (const k of ["score", "lines", "level", "time", "pieces", "best"]) statEls[k] = el.querySelector(`[data-s="${k}"]`);
  const lastStats = {};
  function setStat(k, v) {
    if (lastStats[k] !== v) {
      lastStats[k] = v;
      statEls[k].textContent = v;
    }
  }
  function drawStats() {
    setStat("score", game.score.toLocaleString());
    setStat("lines", String(game.lines));
    setStat("level", String(game.level));
    const s = Math.floor(game.elapsed / 1000);
    setStat("time", `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`);
    setStat("pieces", String(game.pieces));
    setStat("best", Math.max(best.score, game.score).toLocaleString());
  }

  function floatText(title, sub) {
    const s = document.createElement("span");
    s.innerHTML = `${esc(title)}<small>${esc(sub)}</small>`;
    ui.float.replaceChildren(s);
  }

  let toastTimer = 0;
  function toast(msg) {
    let t = document.getElementById("toast");
    if (!t) {
      t = document.createElement("div");
      t.id = "toast";
      t.style.cssText = "position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:60;background:#141925;border:1px solid #2c3749;color:#e7ecf3;padding:9px 14px;border-radius:10px;font-size:.86rem;box-shadow:0 10px 30px rgba(0,0,0,.5);transition:opacity .3s";
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.style.opacity = "1";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.style.opacity = "0"), 2600);
  }

  // -------------------------------------------------------------------------------------------
  // model panel

  function showStatus() {
    ui.auto.setAttribute("aria-pressed", String(ai.enabled));
    ui.auto.classList.toggle("thinking", ai.busy);
    const info = session.info;
    ui.aiModel.innerHTML = info
      ? `<span class="badge">${esc(session.nameOf())}</span>${backendBadge(info)}${info.backend !== "webgpu" ? `<span class="badge warn">CPU: slower decisions</span>` : ""}`
      : `<span class="badge"><span class="dot"></span>${session.status === "loading" ? "model loading…" : "no model loaded"}</span>`;
    el.querySelector('[data-m="dec"]').textContent = String(ai.stats.decisions);
    el.querySelector('[data-m="drop"]').textContent = String(ai.stats.dropped);
    if (!ai.stats.ms.length) el.querySelector('[data-m="ms"]').textContent = ai.busy ? "…" : "–";
  }

  const pct = (p) => `${(p * 100).toFixed(p >= 0.995 ? 0 : 1)}%`;

  function showDecision(d) {
    lastDecision = d;
    const recent = ai.stats.ms.slice(-20);
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const ms = el.querySelector('[data-m="ms"]');
    ms.textContent = fmtMs(d.ms);
    ms.title = `last ${fmtMs(d.ms)} · average ${fmtMs(avg)} over ${recent.length} pieces · ${d.timing?.tokens ?? "?"} tokens in one pass`;
    const sp = el.querySelector('[data-m="spots"]');
    sp.textContent = `${d.spots} → ${d.states}`;
    sp.title = `${d.spots} landing spots, ${d.states} distinct descriptions scored in one batch`;
    const top = d.scored.slice(0, 4);
    ui.spots.innerHTML = top
      .map((s, i) => `<li class="${i === 0 ? "chosen" : ""}"><div class="rt"><span>${esc(s.group[0].summary)}${i === 0 ? " · <b style='color:var(--accent-2)'>chosen</b>" : ""}</span><b>${pct(s.p)}</b></div><div class="pbar"><i style="width:0"></i></div>${i === 0 ? `<div class="desc">${esc(s.text)}</div>` : ""}</li>`)
      .join("");
    requestAnimationFrame(() => ui.spots.querySelectorAll(".pbar i").forEach((b, i) => (b.style.width = `${(top[i].p * 100).toFixed(1)}%`)));
    ui.seen.innerHTML = highlightJSON(JSON.stringify({ state: d.best.text, questions: QUESTION }, null, 2));
    showStatus();
  }

  function addTrail(type, keys, p, ms, lines) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="pc" style="background:${COLORS[type]}"></span><span class="tk">${keys.map((k) => GLYPH[k]).join(" ")}${lines ? ` <span style="color:var(--accent-2)">+${lines}</span>` : ""}</span><span class="pp">${pct(p)}</span><span class="ms">${fmtMs(ms)}</span>`;
    ui.trail.prepend(li);
    while (ui.trail.children.length > 6) ui.trail.lastChild.remove();
  }

  // the keys and the plan strip follow the piece every frame
  let keySig = "";
  function drawKeys() {
    const v = ai.enabled ? ai.values() : null;
    const next = ai.enabled ? ai.next() : null;
    const plan = ai.enabled && ai.plan && ai.plan.pieceId === game.pieceId ? ai.plan : null;
    const sig = `${JSON.stringify(v)}|${next}|${plan ? `${plan.pressed}/${plan.keys.length}` : ai.busy}`;
    if (sig === keySig) return;
    keySig = sig;
    for (const k of KEYS) {
      const b = el.querySelector(`.key[data-k="${k}"]`);
      const val = v?.[k];
      b.classList.toggle("none", val == null);
      b.classList.toggle("next", k === next);
      b.querySelector(".kv").textContent = val == null ? "–" : pct(val);
      b.querySelector(".kb i").style.width = `${val == null ? 0 : (val * 100).toFixed(1)}%`;
    }
    if (plan) {
      ui.plan.innerHTML = plan.keys.map((k, i) => `<kbd class="${i < plan.pressed ? "done" : i === plan.pressed ? "now" : ""}" title="${KEY_NAME[k]}">${GLYPH[k]}</kbd>`).join("") + `<span class="pl-t">toward the ${pct(plan.best.p)} spot</span>`;
    } else {
      ui.plan.innerHTML = `<span class="pl-t">${ai.enabled ? (ai.busy ? "The model is scoring the landing spots…" : "Waiting for the next piece…") : "Auto is off."}</span>`;
    }
  }

  // -------------------------------------------------------------------------------------------
  // the loop: fixed steps, bounded catch-up; stops while the view is hidden

  let last = performance.now();
  let acc = 0;
  let raf = 0;
  function frame(now) {
    raf = 0;
    if (!visible) return;
    let dt = now - last;
    last = now;
    if (dt > 250) dt = 250;
    acc += dt;
    let n = 0;
    while (acc >= STEP && n < 32) {
      tick(STEP);
      acc -= STEP;
      n++;
    }
    if (n === 32) acc = 0;
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
  for (const b of el.querySelectorAll(".game button")) b.addEventListener("click", () => ui.board.focus({ preventScroll: true }));
  if (params.get("touch") === "1") el.querySelector(".touch").style.display = "grid";

  showStatus();
  // read-only handle for tests and recordings; ignoreKeys() stops stray typing from taking over
  window.tetris = { game, ai, start, setAuto, ignoreKeys: (on = true) => (keyboard = !on) };

  return {
    show() {
      visible = true;
      layout();
      last = performance.now();
      if (!raf) raf = requestAnimationFrame(frame);
      if (resumeOnShow && started && game.paused && !game.over) togglePause();
      resumeOnShow = false;
      maybeAutoStart();
    },
    hide() {
      visible = false;
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
