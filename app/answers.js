// Answer cards: one per question, drawn for its type. A yes/no answer is a verdict and a split
// bar, a choice is a ranked list with the unlikely options folded away, and a score is a small
// histogram of its levels with a marker at the expected value. Rendering the same questions again
// updates the cards in place, so bars and columns move to their new values instead of redrawing.

const TYPE_LABEL = { noul: "yes / no", choice: "pick one", score: "scale" };

/** A choice folds options under this probability behind a "more" toggle. */
const FOLD_BELOW = 0.01;

const escapeHTML = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** 0.904 -> "90.4%"; the extremes lose their decimal (100%, 0%). */
const percent = (p) => `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;

/** The question as a title: `field` names in backticks become code. */
function questionHTML(text) {
  return escapeHTML(text || "").replace(/`([^`]+)`/g, "<code>$1</code>");
}

function optionLabel(answer, key, labels) {
  if (labels[key] != null) return labels[key];
  const { legend } = answer;
  if (answer.type === "score" && legend && typeof legend === "object") return legend[key] ?? key;
  return key;
}

// -------------------------------------------------------------------------------------------
// card skeletons: built once per question shape, then only updated

function headHTML(id, answer, question) {
  const title = question.instructions ? questionHTML(question.instructions) : `<code>${escapeHTML(id)}</code>`;
  return [
    `<div class="a-head">`,
    `<p class="a-question">${title}</p>`,
    `<span class="a-type">${TYPE_LABEL[answer.type] || escapeHTML(answer.type)}</span>`,
    `</div>`,
    question.instructions ? `<div class="a-id">${escapeHTML(id)}</div>` : "",
    `<div class="a-verdict"><b class="a-word"></b><span class="a-prob"></span></div>`,
  ].join("");
}

function noulHTML() {
  return [
    `<div class="a-split" role="img"><i class="a-split-yes"></i></div>`,
    `<div class="a-ends"><span class="a-end-yes"></span><span class="a-end-no"></span></div>`,
  ].join("");
}

function choiceHTML(keys) {
  const rows = keys.map(
    (key) =>
      `<li class="a-row" data-k="${escapeHTML(key)}"><span class="a-name"></span><span class="a-track"><i></i></span><span class="a-value"></span></li>`,
  );
  return `<ol class="a-rank">${rows.join("")}</ol><button type="button" class="a-more hidden"></button>`;
}

function scoreHTML(keys) {
  const cells = (make) => keys.map((key) => make(escapeHTML(key))).join("");
  return [
    `<div class="a-scale" style="--levels: ${keys.length}">`,
    cells((k) => `<span class="a-col-value" data-k="${k}"></span>`),
    cells((k) => `<span class="a-col-bar" data-k="${k}"><i></i></span>`),
    `<div class="a-axis"><span class="a-mark"></span></div>`,
    cells((k) => `<span class="a-col-label" data-k="${k}"></span>`),
    `</div>`,
  ].join("");
}

function optionKeys(answer) {
  return answer.type === "noul" ? ["true", "false"] : Object.keys(answer.probabilities || {});
}

/** The card for question `id`, rebuilt only when its type or options change. */
function cardFor(container, id, answer, question) {
  const shape = `${answer.type}|${optionKeys(answer).join("\u0001")}|${question.instructions || ""}`;
  let card = container.querySelector(`:scope > [data-q="${CSS.escape(id)}"]`);
  if (card?.dataset.shape === shape) return card;
  const fresh = document.createElement("article");
  fresh.className = `answer a-${answer.type}`;
  fresh.dataset.q = id;
  fresh.dataset.shape = shape;
  const keys = optionKeys(answer);
  const body = answer.type === "noul" ? noulHTML() : answer.type === "score" ? scoreHTML(keys) : choiceHTML(keys);
  fresh.innerHTML = `${headHTML(id, answer, question)}${body}<div class="a-foot"></div>`;
  if (card) card.replaceWith(fresh);
  else container.appendChild(fresh);
  return fresh;
}

// -------------------------------------------------------------------------------------------
// updates

/** Sets a style on the next frame, so a fresh card animates from zero like an updated one. */
function animate(el, prop, value) {
  requestAnimationFrame(() => (el.style[prop] = value));
}

function updateNoul(card, answer, labels) {
  const yes = answer.noul;
  const isYes = yes >= 0.5;
  card.classList.toggle("is-yes", isYes);
  card.querySelector(".a-word").textContent = isYes ? "Yes" : "No";
  card.querySelector(".a-prob").textContent = percent(isYes ? yes : 1 - yes);
  const split = card.querySelector(".a-split");
  split.setAttribute("aria-label", `P(yes) ${percent(yes)}`);
  animate(split.firstElementChild, "width", `${(yes * 100).toFixed(2)}%`);
  card.querySelector(".a-end-yes").textContent = `${labels.true ?? "yes"} ${percent(yes)}`;
  card.querySelector(".a-end-no").textContent = `${labels.false ?? "no"} ${percent(1 - yes)}`;
}

function updateChoice(card, answer, labels, max) {
  const ranked = Object.entries(answer.probabilities || {}).sort((a, b) => b[1] - a[1]);
  const [topKey, topP] = ranked[0] || ["", 0];
  card.querySelector(".a-word").textContent = optionLabel(answer, answer.choice ?? topKey, labels);
  card.querySelector(".a-prob").textContent = percent(answer.probabilities?.[answer.choice] ?? topP);

  // show the likely options; fold the rest unless the card was expanded (or cap them at `max`)
  const expanded = card.classList.contains("expanded");
  const visible = max ? ranked.slice(0, max) : ranked.filter(([, p], i) => i < 3 || p >= FOLD_BELOW || expanded);
  const folded = ranked.length - visible.length;
  const list = card.querySelector(".a-rank");
  for (const [key, p] of ranked) {
    const row = list.querySelector(`[data-k="${CSS.escape(key)}"]`);
    const label = optionLabel(answer, key, labels);
    const name = row.querySelector(".a-name");
    name.textContent = label;
    name.title = label;
    row.querySelector(".a-value").textContent = percent(p);
    row.classList.toggle("top", key === topKey);
    row.classList.toggle("hidden", !visible.some(([k]) => k === key));
    animate(row.querySelector(".a-track i"), "width", `${(p * 100).toFixed(2)}%`);
    list.appendChild(row); // keep rows in rank order
  }
  const more = card.querySelector(".a-more");
  const foldable = !max && ranked.some(([, p], i) => i >= 3 && p < FOLD_BELOW);
  more.classList.toggle("hidden", !foldable);
  more.textContent = expanded ? "Show fewer" : `${folded} more under ${percent(FOLD_BELOW)}`;
  more.onclick = () => {
    card.classList.toggle("expanded");
    updateChoice(card, answer, labels, max);
  };
}

function updateScore(card, answer, labels) {
  const entries = Object.entries(answer.probabilities || {});
  const n = entries.length;
  const expected = typeof answer.score === "number" ? answer.score : entries.reduce((s, [, p], i) => s + i * p, 0);
  const [topKey] = entries.reduce((best, e) => (e[1] > best[1] ? e : best), entries[0] || ["", 0]);
  const nearest = entries[Math.min(n - 1, Math.max(0, Math.round(expected)))]?.[0] ?? topKey;
  card.querySelector(".a-word").textContent = optionLabel(answer, nearest, labels);
  const mean = card.querySelector(".a-prob");
  mean.textContent = `${expected.toFixed(2)} / ${n - 1}`;
  mean.title = `The expected level, on a scale from 0 to ${n - 1}`;

  const tallest = Math.max(...entries.map(([, p]) => p), 1e-9);
  const cell = (part, key) => card.querySelector(`.a-col-${part}[data-k="${CSS.escape(key)}"]`);
  for (const [key, p] of entries) {
    const label = optionLabel(answer, key, labels);
    for (const part of ["value", "bar", "label"]) cell(part, key).classList.toggle("top", key === topKey);
    cell("value", key).textContent = percent(p);
    Object.assign(cell("label", key), { textContent: label, title: label });
    animate(cell("bar", key).firstElementChild, "height", `${((p / tallest) * 100).toFixed(1)}%`);
  }
  // level i is centered at (i + 0.5) / n of the width
  animate(card.querySelector(".a-mark"), "left", `${(((expected + 0.5) / n) * 100).toFixed(2)}%`);
}

function footHTML(answer) {
  const pills = [];
  if (answer.confidence != null) {
    pills.push(
      `<span class="a-pill" title="How sure the model is of this answer (1 = certain)">confidence <b>${answer.confidence.toFixed(2)}</b><i style="--v: ${(answer.confidence * 100).toFixed(0)}%"></i></span>`,
    );
  }
  if (answer.action?.act_probability != null) {
    const act = answer.action.act_probability;
    pills.push(
      `<span class="a-pill" title="Laya's own estimate that acting on this answer is safe">act <b>${act.toFixed(2)}</b><i style="--v: ${(act * 100).toFixed(0)}%"></i></span>`,
    );
  }
  return pills.join("");
}

/**
 * Renders or updates the answers of `response` into `container`. `questions` supplies each
 * question's text; `labels[id]` can rename options; `max` caps a choice to its likeliest options.
 */
export function renderAnswers(container, response, questions = {}, { labels = {}, max = 0 } = {}) {
  container.classList.add("answers");
  const answers = response?.answers || {};
  const seen = new Set(Object.keys(answers));
  for (const [id, answer] of Object.entries(answers)) {
    const card = cardFor(container, id, answer, questions[id] || {});
    const names = labels[id] || {};
    if (answer.type === "noul") updateNoul(card, answer, names);
    else if (answer.type === "score") updateScore(card, answer, names);
    else updateChoice(card, answer, names, max);
    card.querySelector(".a-foot").innerHTML = footHTML(answer);
  }
  for (const card of [...container.children]) if (card.dataset.q && !seen.has(card.dataset.q)) card.remove();
}
