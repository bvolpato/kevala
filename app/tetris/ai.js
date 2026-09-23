// The auto player. When a piece appears, the code lists every spot its controller can reach
// (the game's own collision and SRS kick code), measures each outcome, and rejects worse boards.
// The model scores the remaining distinct outcomes with one batched yes/no question per outcome.
// The piece then plays the keys toward the highest P(yes), one press at a time.
//
// Keys are how the choice shows up on screen: each key's value is the best P(yes) among the
// shortlisted representatives that key leads toward (turn first, then slide, then drop).
// While a piece moves, the next piece's promising spots are scored on its expected board. The
// answer is used only if that board and piece are exactly the ones asked about.

import { enumeratePlacements, measure, rotated, shifted, place, spawnPiece, pieceFits, dropDistance } from "./engine.js";

export const KEYS = ["left", "right", "rotate", "drop"];
export const GLYPH = { left: "←", right: "→", rotate: "↻", drop: "⤓" };
export const KEY_NAME = { left: "Left", right: "Right", rotate: "Turn", drop: "Drop" };

export const QUESTION = {
  clean: {
    type: "noul",
    instructions: "Does the stack look clean after this move?",
    criteria: { true: "clean: no holes, low and flat", false: "messy: holes under blocks, tall or jagged" },
  },
};

const NUM = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const num = (n) => NUM[n] ?? String(n);
const rows = (n) => `${num(n)} ${n === 1 ? "row" : "rows"}`;
const SHORTLIST_MARGIN = 0.75;
const SHORTLIST_SIZE = 4;

function value(m) {
  // Favor completed rows and avoid buried cells; height and roughness break safer-move ties.
  return 8 * m.lines - 9 * m.newHoles - 2 * m.holesAfter - 0.5 * m.totalHeight - 0.5 * m.bumpiness - 0.3 * m.maxHeight;
}

/**
 * Where a value sits among the distinct values every option produces (lower is better for
 * bumpiness and height): "lo" in the better part, "hi" in the worse part, "mid" otherwise.
 * This is a comparison computed in code and stated in words.
 */
function tier(v, values) {
  const u = [...new Set(values)].sort((a, b) => a - b);
  if (u.length === 1) return "mid";
  const f = u.indexOf(v) / (u.length - 1);
  return f <= 0.4 ? "lo" : f >= 0.7 ? "hi" : "mid";
}

/** Plain-words description of one outcome, given the outcomes of all the options. */
export function describe(m, all) {
  const s = [];
  s.push(m.holesAfter
    ? `The stack has ${num(m.holesAfter)} ${m.holesAfter === 1 ? "hole" : "holes"} after this move.`
    : "The stack has no holes after this move.");
  if (m.holesAfter > m.holesBefore) s.push(`That is ${num(m.holesAfter - m.holesBefore)} more than before.`);
  if (m.holesAfter < m.holesBefore) s.push(`That is ${num(m.holesBefore - m.holesAfter)} fewer than before.`);
  if (m.lines) s.push(`It completes ${rows(m.lines)}.`);
  const b = tier(m.bumpiness, all.map((x) => x.bumpiness));
  s.push({ lo: "The surface ends up flatter than with most other moves.", mid: "The surface ends up about as flat as with most other moves.", hi: "The surface ends up bumpier than with most other moves." }[b]);
  const h = tier(m.maxHeight, all.map((x) => x.maxHeight));
  if (h === "lo") s.push("The stack stays lower than with most other moves.");
  if (h === "hi") s.push("The stack grows taller than with most other moves.");
  return s.join(" ");
}

/** A few words for the panel. */
export function summary(m, all) {
  const b = tier(m.bumpiness, all.map((x) => x.bumpiness));
  return [m.lines ? `clears ${m.lines}` : null, m.holesAfter ? `${m.holesAfter} hole${m.holesAfter > 1 ? "s" : ""}` : "no holes", { lo: "flatter", mid: "even", hi: "bumpier" }[b]].filter(Boolean).join(" · ");
}

/** Measure all controller-reachable landings and ask the model about the safest distinct outcomes. */
export function candidates(board, piece) {
  const items = enumeratePlacements(board, piece).map((p) => {
    const m = measure(board, p.final);
    return { ...p, m, value: value(m) };
  });
  const all = items.map((i) => i.m);
  const grouped = new Map();
  for (const it of items) {
    it.text = describe(it.m, all);
    it.summary = summary(it.m, all);
    if (!grouped.has(it.text)) grouped.set(it.text, []);
    grouped.get(it.text).push(it);
  }
  for (const g of grouped.values()) g.sort((a, b) => b.value - a.value || a.cost - b.cost);
  const ranked = [...grouped].sort((a, b) => b[1][0].value - a[1][0].value || a[1][0].cost - b[1][0].cost);
  const best = ranked[0]?.[1][0].value;
  const groups = new Map(ranked.filter(([, g]) => best - g[0].value <= SHORTLIST_MARGIN).slice(0, SHORTLIST_SIZE));
  return { items, groups, texts: [...groups.keys()] };
}

function reachable(board, pose, spot) {
  let p = pose;
  for (const dir of spot.rotations) {
    p = rotated(board, p, dir);
    if (!p) return false;
  }
  while (p.x !== spot.target.x) {
    p = shifted(board, p, Math.sign(spot.target.x - p.x));
    if (!p) return false;
  }
  return p.rot === spot.final.rot && p.y + dropDistance(board, p) === spot.final.y;
}

/** The next key toward a spot from pose `p`: turn first, then slide, then drop. */
export function nextKey(p, spot) {
  if (p.rot !== spot.target.rot) return "rotate";
  if (p.x > spot.target.x) return "left";
  if (p.x < spot.target.x) return "right";
  return "drop";
}

/** The key sequence toward a spot, simulated with the game's own moves, for the plan strip. */
export function keysTo(board, p, spot) {
  const keys = [];
  let q = p;
  for (let i = 0; i < 3 && q.rot !== spot.target.rot; i++) {
    const r = rotated(board, q, 1);
    if (!r) break;
    q = r;
    keys.push("rotate");
  }
  for (let i = 0; i < 10 && q.x !== spot.target.x; i++) {
    const dir = q.x < spot.target.x ? 1 : -1;
    const n = shifted(board, q, dir);
    if (!n) break;
    q = n;
    keys.push(dir < 0 ? "left" : "right");
  }
  keys.push("drop");
  return keys;
}

/** Per key: the best P(yes) among the shortlisted representatives from pose `p`. */
export function keyValues(p, scored) {
  const v = Object.fromEntries(KEYS.map((k) => [k, null]));
  for (const s of scored) {
    const k = nextKey(p, s.group[0]);
    if (v[k] === null || s.p > v[k]) v[k] = s.p;
  }
  return v;
}

/**
 * P(true) for the question. Kev rounds `answers` to 2 places like upstream Kev, which ties many
 * options; its `raw_probabilities` keep full precision, so use the raw entry that matches.
 */
function pTrue(r) {
  const p = r.answers.clean.noul;
  const raw = r.raw_probabilities?.clean;
  if (!Array.isArray(raw) || raw.length !== 2) return p;
  return Math.abs(raw[1] - p) <= Math.abs(raw[0] - p) ? raw[1] : raw[0];
}

/** Where a new piece of `type` appears on `board`, as Game.spawn places it; null on a top-out. */
function spawnPose(board, type) {
  const p = spawnPiece(type);
  if (!pieceFits(board, p)) return null;
  return shifted(board, p, 0, 1) || p;
}

/** Identifies a moment of the game: the settled board plus the falling piece's pose. */
const momentKey = (board, p) => `${board.join("")}|${p.type}${p.rot}:${p.x},${p.y}`;

export const SPEEDS = {
  chill: { label: "Chill", gap: 260, dropGap: 30 },
  normal: { label: "Normal", gap: 65, dropGap: 16 },
  turbo: { label: "Turbo", gap: 25, dropGap: 8 },
};

/**
 * Drives a Game: asks the model when a piece appears, then presses the keys toward the chosen
 * spot while gravity keeps running.
 *
 * Every request carries the game epoch and piece id; a result is used only if both still match
 * and Auto is still on. At most one request is in flight.
 */
export class AutoPlayer {
  constructor(game, { getModel, onDecision, onPress, onStatus, onError }) {
    this.game = game;
    this.getModel = getModel;
    this.onDecision = onDecision;
    this.onPress = onPress;
    this.onStatus = onStatus;
    this.onError = onError;
    this.enabled = false;
    this.epoch = 0;
    this.inflight = null;
    this.ahead = null; // the next piece's question, asked while this one moves
    this.plan = null;
    this.speed = "normal";
    this.timer = 0;
    this.stats = AutoPlayer.#freshStats();
  }

  static #freshStats() {
    return { decisions: 0, dropped: 0, ahead: 0, ms: [], states: 0, spots: 0 };
  }

  get busy() {
    return !!this.inflight;
  }

  setEnabled(on) {
    if (on === this.enabled) return;
    this.enabled = on;
    this.epoch++;
    this.plan = null;
    this.ahead = null;
    this.game.softDrop = false;
    this.onStatus?.();
    if (on) this.request();
  }

  /** A human input arrived: give control back on the same board. */
  takeover() {
    if (!this.enabled) return false;
    this.setEnabled(false);
    return true;
  }

  /** The game restarted: forget everything tied to the old board. */
  reset() {
    this.epoch++;
    this.plan = null;
    this.ahead = null;
    this.game.softDrop = false;
    this.stats = AutoPlayer.#freshStats();
    this.onStatus?.();
  }

  onSpawn() {
    this.plan = null;
    this.game.softDrop = false;
    if (this.enabled) this.request();
  }

  /**
   * Gets the model's answer for the current piece. When the answer was already asked for while
   * the previous piece was moving, and the board and piece are exactly the ones it was asked
   * about, that answer is used; otherwise the model is asked now.
   */
  request() {
    const g = this.game;
    const kevala = this.getModel();
    if (!this.enabled || this.inflight || !kevala || g.over || g.paused || !g.piece) return;
    const ahead = this.ahead;
    this.ahead = null;
    const fits = ahead && ahead.epoch === this.epoch && ahead.key === momentKey(g.board, g.piece);
    const job = fits ? ahead : this.#ask(kevala, g.board, g.piece);
    this.#settle(kevala, job, { epoch: this.epoch, pieceId: g.pieceId }, fits);
  }

  /** Starts one batched question for the promising distinct outcomes of `piece` on `board`. */
  #ask(kevala, board, piece) {
    const c = candidates(board, piece);
    const job = { key: momentKey(board, piece), epoch: this.epoch, c, t0: performance.now(), ms: 0 };
    job.answer = kevala.decideMany(c.texts.map((state) => ({ state, questions: QUESTION }))).then((res) => {
      job.ms = performance.now() - job.t0;
      return res;
    });
    job.answer.catch(() => {}); // a job that is never used must not report an unhandled error
    return job;
  }

  /** Waits for a job's answer, turns it into the plan for the current piece, and looks ahead. */
  #settle(kevala, job, tag, ahead) {
    this.inflight = tag;
    this.onStatus?.();
    job.answer
      .then((res) => {
        this.inflight = null;
        const g = this.game;
        if (tag.epoch !== this.epoch || tag.pieceId !== g.pieceId || !this.enabled || g.over) {
          this.stats.dropped++;
          this.onStatus?.();
          this.request(); // the piece changed while the model thought: ask about the current one
          return;
        }
        const { c } = job;
        const scored = c.texts.map((text, i) => ({ text, p: pTrue(res[i]), group: c.groups.get(text).filter((spot) => reachable(g.board, g.piece, spot)) }))
          .filter((s) => s.group.length);
        if (!scored.length) {
          this.stats.dropped++;
          this.request();
          return;
        }
        scored.sort((a, b) => b.p - a.p || b.group[0].value - a.group[0].value);
        const best = scored[0];
        const spot = best.group[0];
        this.plan = { pieceId: tag.pieceId, best, spot, scored, keys: keysTo(g.board, g.piece, spot), pressed: 0 };
        this.timer = 0;
        const st = this.stats;
        st.decisions++;
        if (ahead) st.ahead++;
        st.ms.push(job.ms);
        if (st.ms.length > 200) st.ms.shift();
        st.states += c.texts.length;
        st.spots += c.items.length;
        this.onDecision?.({ piece: g.piece.type, ms: job.ms, ahead, timing: res[0].timing, spots: c.items.length, states: c.texts.length, scored, best, keys: this.plan.keys });
        this.onStatus?.();
        this.#lookAhead(kevala);
      })
      .catch((e) => {
        this.inflight = null;
        this.setEnabled(false);
        this.onError?.(e);
      });
  }

  /**
   * The chosen spot fixes the board the next piece will fall on, and the queue says which piece
   * it is: ask about it now, while this piece moves, so its answer is ready when it appears.
   */
  #lookAhead(kevala) {
    const g = this.game;
    const next = g.queue[0];
    if (!this.plan || !next) return;
    const { board } = place(g.board, this.plan.spot.final);
    const pose = spawnPose(board, next);
    if (pose) this.ahead = this.#ask(kevala, board, pose);
  }

  /** The key values from where the piece is now, or null before the model has answered. */
  values() {
    if (!this.plan || this.plan.pieceId !== this.game.pieceId || !this.game.piece) return null;
    return keyValues(this.game.piece, this.plan.scored);
  }

  /** The key the plan presses next. */
  next() {
    if (!this.plan || this.plan.pieceId !== this.game.pieceId || !this.game.piece) return null;
    if (this.plan.dropping) return "drop";
    return nextKey(this.game.piece, this.plan.spot);
  }

  /** Presses the next key of the plan when its time has come; called from the game loop. */
  tick(dt) {
    const g = this.game;
    const plan = this.plan;
    if (!this.enabled || !plan || plan.pieceId !== g.pieceId || !g.active) return;
    if (plan.dropping) {
      this.timer += dt;
      if (this.timer >= SPEEDS[this.speed].dropGap) {
        this.timer = 0;
        if (!g.stepDown() || g.onGround) g.hardDrop();
      }
      return;
    }
    this.timer += dt;
    if (this.timer < SPEEDS[this.speed].gap) return;
    this.timer = 0;
    const key = this.next();
    let ok = true;
    if (key === "rotate") ok = g.rotate(1);
    else if (key === "left") ok = g.move(-1);
    else if (key === "right") ok = g.move(1);
    if (!ok) {
      // Gravity changed the path while the piece moved; score landings from its current pose.
      this.plan = null;
      this.ahead = null;
      this.request();
      return;
    }
    plan.pressed++;
    this.onPress?.(key);
    if (key === "drop") {
      plan.dropping = true;
      this.timer = 0;
    }
  }
}
