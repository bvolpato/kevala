// Tetris game logic: board, pieces, SRS rotation with wall kicks, seven-bag, hold, lock delay,
// scoring and levels. No DOM here; the page drives it with step(dt) and the input methods.
//
// Coordinates: x grows to the right, y grows downward, row 0 is the top of the hidden spawn area.

export const W = 10;
export const VISIBLE = 20;
export const HIDDEN = 4;
export const H = VISIBLE + HIDDEN;
export const TYPES = ["I", "O", "T", "S", "Z", "J", "L"];

export const LOCK_DELAY = 500; // ms on the ground before a piece locks
export const MAX_RESETS = 15; // moves/rotations that may restart the lock timer
export const SOFT_DROP_MS = 30; // ms per row while soft dropping

// state 0 of every piece inside its bounding box; the other states are rotations of it
const SPAWN = {
  I: { n: 4, cells: [[0, 1], [1, 1], [2, 1], [3, 1]] },
  O: { n: 2, cells: [[0, 0], [1, 0], [0, 1], [1, 1]] },
  T: { n: 3, cells: [[1, 0], [0, 1], [1, 1], [2, 1]] },
  S: { n: 3, cells: [[1, 0], [2, 0], [0, 1], [1, 1]] },
  Z: { n: 3, cells: [[0, 0], [1, 0], [1, 1], [2, 1]] },
  J: { n: 3, cells: [[0, 0], [0, 1], [1, 1], [2, 1]] },
  L: { n: 3, cells: [[2, 0], [0, 1], [1, 1], [2, 1]] },
};

/** SHAPES[type][rot] = [[dx, dy] x4], rotation states 0, R, 2, L (clockwise order). */
export const SHAPES = {};
for (const t of TYPES) {
  const { n, cells } = SPAWN[t];
  const states = [cells];
  for (let r = 1; r < 4; r++) states.push(states[r - 1].map(([x, y]) => [n - 1 - y, x]));
  SHAPES[t] = states.map((s) => s.slice().sort((a, b) => a[1] - b[1] || a[0] - b[0]));
}
export const BOX = Object.fromEntries(TYPES.map((t) => [t, SPAWN[t].n]));

// SRS kick tests, written as in the SRS reference (y up) and flipped to y down below
const KICKS_JLSTZ = {
  "0>1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "1>0": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "1>2": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "2>1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "2>3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "3>2": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "3>0": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "0>3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
};
const KICKS_I = {
  "0>1": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "1>0": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "1>2": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  "2>1": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
  "2>3": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "3>2": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "3>0": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
  "0>3": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
};
const flip = (table) => Object.fromEntries(Object.entries(table).map(([k, v]) => [k, v.map(([x, y]) => [x, -y])]));
const KICKS = { I: flip(KICKS_I), JLSTZ: flip(KICKS_JLSTZ) };

export function kicksFor(type, from, to) {
  if (type === "O") return [[0, 0]]; // O never kicks: its rotations are the same four cells
  return (type === "I" ? KICKS.I : KICKS.JLSTZ)[`${from}>${to}`];
}

/** Spawn pose: centered, in the two hidden rows just above the visible field. */
export function spawnPiece(type) {
  const x = type === "O" ? 4 : 3;
  return { type, rot: 0, x, y: HIDDEN - 2 };
}

export function newBoard() {
  return new Uint8Array(W * H);
}

export function fits(board, type, rot, x, y) {
  for (const [dx, dy] of SHAPES[type][rot]) {
    const cx = x + dx;
    const cy = y + dy;
    if (cx < 0 || cx >= W || cy < 0 || cy >= H) return false;
    if (board[cy * W + cx]) return false;
  }
  return true;
}

export const pieceFits = (board, p) => fits(board, p.type, p.rot, p.x, p.y);

export function pieceCells(p) {
  return SHAPES[p.type][p.rot].map(([dx, dy]) => [p.x + dx, p.y + dy]);
}

/** Rotates with SRS kicks. dir: +1 clockwise, -1 counter-clockwise. Returns the new pose or null. */
export function rotated(board, p, dir) {
  const to = (p.rot + (dir > 0 ? 1 : 3)) % 4;
  const tests = kicksFor(p.type, p.rot, to);
  for (let i = 0; i < tests.length; i++) {
    const [kx, ky] = tests[i];
    if (fits(board, p.type, to, p.x + kx, p.y + ky)) return { ...p, rot: to, x: p.x + kx, y: p.y + ky, kick: i };
  }
  return null;
}

export function shifted(board, p, dx, dy = 0) {
  return fits(board, p.type, p.rot, p.x + dx, p.y + dy) ? { ...p, x: p.x + dx, y: p.y + dy } : null;
}

export function dropDistance(board, p) {
  let d = 0;
  while (fits(board, p.type, p.rot, p.x, p.y + d + 1)) d++;
  return d;
}

/** Writes the piece into a copy of the board and clears full rows. */
export function place(board, p) {
  const b = board.slice();
  for (const [x, y] of pieceCells(p)) b[y * W + x] = TYPES.indexOf(p.type) + 1;
  const rows = [];
  for (let y = 0; y < H; y++) {
    let full = true;
    for (let x = 0; x < W; x++) if (!b[y * W + x]) { full = false; break; }
    if (full) rows.push(y);
  }
  if (rows.length) {
    const out = new Uint8Array(W * H);
    let dst = H - 1;
    for (let y = H - 1; y >= 0; y--) {
      if (rows.includes(y)) continue;
      out.set(b.subarray(y * W, (y + 1) * W), dst * W);
      dst--;
    }
    return { board: out, rows };
  }
  return { board: b, rows };
}

// ---------------------------------------------------------------------------------------------
// Measurements the auto player turns into words. None of this ranks moves; it only counts.

export function columnHeights(board) {
  const h = new Array(W).fill(0);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      if (board[y * W + x]) {
        h[x] = H - y;
        break;
      }
    }
  }
  return h;
}

/** Empty cells with a filled cell somewhere above them in the same column. */
export function countHoles(board) {
  let holes = 0;
  for (let x = 0; x < W; x++) {
    let roof = false;
    for (let y = 0; y < H; y++) {
      if (board[y * W + x]) roof = true;
      else if (roof) holes++;
    }
  }
  return holes;
}

/** Facts about the board after a placement, relative to the board before it. */
export function measure(before, p) {
  const { board, rows } = place(before, p);
  const h = columnHeights(board);
  const hb = columnHeights(before);
  let bump = 0;
  let bumpBefore = 0;
  let maxStep = 0;
  for (let x = 0; x < W - 1; x++) {
    const s = Math.abs(h[x] - h[x + 1]);
    bump += s;
    bumpBefore += Math.abs(hb[x] - hb[x + 1]);
    maxStep = Math.max(maxStep, s);
  }
  const sum = (a) => a.reduce((x, y) => x + y, 0);
  // wells: columns lower than both neighbours (walls count as tall)
  const wells = [];
  for (let x = 0; x < W; x++) {
    const l = x === 0 ? Infinity : h[x - 1];
    const r = x === W - 1 ? Infinity : h[x + 1];
    const depth = Math.min(l, r) - h[x];
    if (depth >= 3 && depth !== Infinity) wells.push({ x, depth });
  }
  const cells = pieceCells(p);
  const lowest = Math.max(...cells.map(([, y]) => y));
  const holesBefore = countHoles(before);
  const holesAfter = countHoles(board);
  return {
    board,
    lines: rows.length,
    holesBefore,
    holesAfter,
    newHoles: Math.max(0, holesAfter - holesBefore),
    filledHoles: Math.max(0, holesBefore - holesAfter),
    maxHeight: Math.max(...h),
    maxHeightBefore: Math.max(...hb),
    bumpiness: bump,
    bumpinessBefore: bumpBefore,
    maxStep,
    totalHeight: sum(h),
    totalHeightBefore: sum(hb),
    minHeightBefore: Math.min(...hb),
    wells,
    landingHeight: H - lowest, // 1 = the piece touches the floor
    heights: h,
  };
}

/**
 * Every distinct place the piece can end up by rotating from its current pose, sliding sideways
 * and dropping, found with the real collision and kick code. Each placement carries the inputs
 * that reach it.
 */
export function enumeratePlacements(board, piece) {
  // clockwise only, so the auto player reaches every spot with the one turn key it presses
  const seqs = [[], [1], [1, 1], [1, 1, 1]];
  const seen = new Map();
  for (const seq of seqs) {
    let p = { ...piece };
    let ok = true;
    for (const d of seq) {
      const r = rotated(board, p, d);
      if (!r) { ok = false; break; }
      p = r;
    }
    if (!ok) continue;
    const add = (q, dx) => {
      const land = { type: q.type, rot: q.rot, x: q.x, y: q.y + dropDistance(board, q) };
      const key = pieceCells(land).map(([x, y]) => y * W + x).sort((a, b) => a - b).join(",");
      const cost = seq.length + Math.abs(dx);
      const prev = seen.get(key);
      if (!prev || cost < prev.cost) seen.set(key, { key, rotations: seq, dx, target: { rot: q.rot, x: q.x }, final: land, cost });
    };
    add(p, 0);
    for (const dir of [-1, 1]) {
      let q = p;
      let dx = 0;
      for (;;) {
        const n = shifted(board, q, dir);
        if (!n) break;
        q = n;
        dx += dir;
        add(q, dx);
      }
    }
  }
  return [...seen.values()];
}

// ---------------------------------------------------------------------------------------------

/** Seeded PRNG (mulberry32). */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seconds per row at a level, from the usual guideline curve. */
export function gravitySeconds(level) {
  const l = Math.min(level, 20);
  return Math.pow(0.8 - (l - 1) * 0.007, l - 1);
}

const LINE_POINTS = [0, 100, 300, 500, 800];
const TSPIN_POINTS = [400, 800, 1200, 1600];

export class Game {
  constructor(seed = 1) {
    this.listeners = new Map();
    this.reset(seed);
  }

  on(ev, fn) {
    if (!this.listeners.has(ev)) this.listeners.set(ev, []);
    this.listeners.get(ev).push(fn);
  }

  emit(ev, data) {
    for (const fn of this.listeners.get(ev) || []) fn(data);
  }

  reset(seed) {
    this.seed = seed >>> 0;
    this.random = rng(this.seed);
    this.board = newBoard();
    this.queue = [];
    this.hold = null;
    this.holdUsed = false;
    this.score = 0;
    this.lines = 0;
    this.level = 1;
    this.combo = -1;
    this.b2b = false;
    this.pieces = 0;
    this.pieceId = 0;
    this.elapsed = 0;
    this.over = false;
    this.paused = false;
    this.softDrop = false;
    this.fall = 0;
    this.lockTimer = 0;
    this.resets = 0;
    this.lowest = 0;
    this.lastRotate = false;
    this.lastClear = null;
    this.stats = { singles: 0, doubles: 0, triples: 0, tetrises: 0, tspins: 0 };
    this.refill();
    this.spawn(this.queue.shift());
  }

  refill() {
    while (this.queue.length < 7) {
      const bag = TYPES.slice();
      for (let i = bag.length - 1; i > 0; i--) {
        const j = Math.floor(this.random() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
      }
      this.queue.push(...bag);
    }
  }

  spawn(type) {
    this.refill();
    const p = spawnPiece(type);
    this.pieceId++;
    this.fall = 0;
    this.lockTimer = 0;
    this.resets = 0;
    this.lastRotate = false;
    if (!pieceFits(this.board, p)) {
      this.piece = p;
      this.gameOver("block out");
      return;
    }
    // appear in the visible field right away when there is room
    const down = shifted(this.board, p, 0, 1);
    this.piece = down || p;
    this.lowest = this.piece.y;
    this.emit("spawn", { pieceId: this.pieceId, piece: this.piece });
  }

  gameOver(reason) {
    this.over = true;
    this.overReason = reason;
    this.emit("gameover", { reason, score: this.score, lines: this.lines });
  }

  get gravityMs() {
    // a gentle ramp on top of the level curve: up to 1.6x faster after ten minutes
    const ramp = 1 + 0.6 * Math.min(1, this.elapsed / 600000);
    return (gravitySeconds(this.level) * 1000) / ramp;
  }

  get onGround() {
    return !pieceFits(this.board, { ...this.piece, y: this.piece.y + 1 });
  }

  /** A successful move or rotation restarts the lock timer, a bounded number of times. */
  moved() {
    if (this.onGround && this.resets < MAX_RESETS) {
      this.lockTimer = 0;
      this.resets++;
    }
  }

  move(dx) {
    if (!this.active) return false;
    const n = shifted(this.board, this.piece, dx);
    if (!n) return false;
    this.piece = n;
    this.lastRotate = false;
    this.moved();
    return true;
  }

  rotate(dir) {
    if (!this.active) return false;
    const n = rotated(this.board, this.piece, dir);
    if (!n) return false;
    this.piece = { type: n.type, rot: n.rot, x: n.x, y: n.y };
    this.lastRotate = true;
    this.lastKick = n.kick;
    this.moved();
    return true;
  }

  /** One row down, as a soft drop step. */
  stepDown(points = 1) {
    if (!this.active) return false;
    const n = shifted(this.board, this.piece, 0, 1);
    if (!n) return false;
    this.piece = n;
    this.lastRotate = false;
    this.score += points;
    this.touchLowest();
    return true;
  }

  touchLowest() {
    if (this.piece.y > this.lowest) {
      this.lowest = this.piece.y;
      this.resets = 0;
      this.lockTimer = 0;
    }
  }

  hardDrop() {
    if (!this.active) return false;
    const d = dropDistance(this.board, this.piece);
    this.piece = { ...this.piece, y: this.piece.y + d };
    if (d) this.lastRotate = false;
    this.score += 2 * d;
    this.emit("harddrop", { piece: this.piece, rows: d });
    this.lock();
    return true;
  }

  holdPiece() {
    if (!this.active || this.holdUsed) return false;
    const cur = this.piece.type;
    const next = this.hold ?? this.queue.shift();
    this.hold = cur;
    this.holdUsed = true;
    this.spawn(next);
    this.emit("hold", { pieceId: this.pieceId });
    return true;
  }

  get active() {
    return !this.over && !this.paused && !!this.piece;
  }

  /** T-spin by the three-corner rule: last action a rotation, three of four corners filled. */
  tspin() {
    const p = this.piece;
    if (p.type !== "T" || !this.lastRotate) return false;
    let n = 0;
    for (const [dx, dy] of [[0, 0], [2, 0], [0, 2], [2, 2]]) {
      const x = p.x + dx;
      const y = p.y + dy;
      if (x < 0 || x >= W || y >= H || this.board[y * W + x]) n++;
    }
    return n >= 3;
  }

  lock() {
    const p = this.piece;
    const spin = this.tspin();
    const cells = pieceCells(p);
    const { board, rows } = place(this.board, p);
    this.board = board;
    this.pieces++;
    const n = rows.length;
    let pts = 0;
    let hard = false;
    if (spin) {
      pts = TSPIN_POINTS[n] * this.level;
      hard = n > 0;
      this.stats.tspins++;
    } else if (n) {
      pts = LINE_POINTS[n] * this.level;
      hard = n === 4;
    }
    if (n) {
      if (hard && this.b2b) pts = Math.floor(pts * 1.5);
      this.b2b = hard;
      this.combo++;
      if (this.combo > 0) pts += 50 * this.combo * this.level;
      this.stats[["", "singles", "doubles", "triples", "tetrises"][n]]++;
    } else {
      this.combo = -1;
    }
    this.score += pts;
    this.lines += n;
    this.level = 1 + Math.floor(this.lines / 10);
    this.lastClear = n || spin ? { rows, n, spin, points: pts, at: this.elapsed } : this.lastClear;
    this.holdUsed = false;
    this.emit("lock", { pieceId: this.pieceId, piece: p, rows, points: pts, spin });
    if (cells.every(([, y]) => y < HIDDEN)) {
      this.piece = null;
      this.gameOver("lock out");
      return;
    }
    this.spawn(this.queue.shift());
  }

  /** Advances time by dt ms: gravity, soft drop and lock delay. */
  step(dt) {
    if (this.over || this.paused || !this.piece) return;
    this.elapsed += dt;
    const g = this.softDrop ? Math.min(this.gravityMs, SOFT_DROP_MS) : this.gravityMs;
    this.fall += dt;
    while (this.fall >= g) {
      this.fall -= g;
      const n = shifted(this.board, this.piece, 0, 1);
      if (!n) {
        this.fall = 0;
        break;
      }
      this.piece = n;
      this.lastRotate = false;
      if (this.softDrop) this.score += 1;
      this.touchLowest();
    }
    if (this.onGround) {
      this.lockTimer += dt;
      if (this.lockTimer >= LOCK_DELAY || this.resets >= MAX_RESETS) this.lock();
    } else {
      this.lockTimer = 0;
    }
  }

  ghost() {
    if (!this.piece) return null;
    return { ...this.piece, y: this.piece.y + dropDistance(this.board, this.piece) };
  }
}
