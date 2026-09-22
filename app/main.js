// The site shell: header tabs, the model menu, and a hash router that keeps every visited view
// alive. Views are ES modules loaded on first visit; hiding one calls its hide() so games pause
// and timers stop, and the model keeps loading in the session whatever the tab.

import { session, MODELS, MODEL_NOTES, LOCAL, SHOT, FROM } from "./session.js";
import { esc, fmtBytes, fmtMs, backendLabel, REPO } from "./ui.js";

/** A view with a tab in the header; without a label it has no tab. */
const defineRoute = (id, label, title) => ({ id, label, title, load: () => import(`./views/${id}.js`) });

const ROUTES = [
  defineRoute("home", null, "kevala · decision models that run in the page"),
  defineRoute("playground", "Playground", "Playground · kevala"),
  defineRoute("tetris", "Tetris", "Tetris played by a decision model · kevala"),
  defineRoute("guardrail", "Guardrail", "Prompt guardrail · kevala"),
  defineRoute("inbox", "Inbox", "Inbox triage · kevala"),
  defineRoute("how", "How it works", "How kevala works"),
];

const LOGO = [
  `<svg viewBox="0 0 32 32" aria-hidden="true">`,
  `<defs><linearGradient id="wg" x1="0" y1="0" x2="1" y2="1">`,
  `<stop offset="0" stop-color="#7aa2ff"/><stop offset=".55" stop-color="#45e0c0"/><stop offset="1" stop-color="#c592ff"/>`,
  `</linearGradient></defs>`,
  `<rect x="1" y="1" width="30" height="30" rx="9" fill="#0f131b" stroke="url(#wg)" stroke-width="2"/>`,
  `<path d="M11.5 7.5v17M21 12.5l-9 6.5M15.6 16.4l5.9 8.1" fill="none" stroke="url(#wg)" stroke-width="2.6"` +
    ` stroke-linecap="round" stroke-linejoin="round"/>`,
  `</svg>`,
].join("");

const CARET =
  `<svg class="mc-caret" viewBox="0 0 10 6" aria-hidden="true">` +
  `<path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;

// Header

const header = document.querySelector("header.nav");
const navLinks = ROUTES.filter((r) => r.label)
  .map((r) => `<a href="#/${r.id}" data-route="${r.id}">${esc(r.label)}</a>`)
  .join("");
const devBadge = LOCAL && !SHOT ? `<span class="badge warn" title="Loading packs from this server's tmp/">dev packs</span>` : "";
const chipParts = `<span class="mc-dot"></span><span class="mc-name"></span><span class="mc-state"></span>${CARET}`;

header.innerHTML = `<div class="wrap">
  <a class="brand" href="#/">${LOGO}<span>kevala</span></a>
  <nav class="tabs-nav" aria-label="Sections">${navLinks}</nav>
  <span class="spacer"></span>
  ${devBadge}
  <div class="mm">
    <button type="button" class="mchip" aria-haspopup="dialog" aria-expanded="false">${chipParts}</button>
    <div class="mpanel hidden" role="dialog" aria-label="Model"></div>
  </div>
  <a class="nav-gh" href="${REPO}" title="Source on GitHub">GitHub</a>
</div><div class="gprog hidden"><i></i></div>`;

const chip = header.querySelector(".mchip");
const panel = header.querySelector(".mpanel");
const globalProgress = header.querySelector(".gprog");

function chipStatus(s) {
  if (s.status === "ready") return backendLabel(s.info);
  if (s.status === "loading") return s.progress && !s.progress.indet ? `${Math.floor(s.progress.frac * 100)}%` : "loading";
  if (s.status === "error") return "error";
  return "load";
}

function renderChip(s) {
  chip.dataset.state = s.status;
  chip.querySelector(".mc-name").textContent = s.nameOf();
  chip.querySelector(".mc-state").textContent = chipStatus(s);
  globalProgress.classList.toggle("hidden", s.status !== "loading");
  if (s.status === "loading") {
    globalProgress.classList.toggle("indet", !!s.progress?.indet);
    globalProgress.firstElementChild.style.width = `${Math.round((s.progress?.frac || 0) * 100)}%`;
  }
}

function modelOption(s, id) {
  const note = MODEL_NOTES[id];
  const active = s.model === id;
  const { hosted, pack, download } = MODELS[id];
  let badge = `<span class="badge faint-b">${fmtBytes(hosted && FROM === "pack" ? pack : download)} download</span>`;
  if (active && s.status === "ready") badge = `<span class="badge good">loaded</span>`;
  else if (s.cached[id]) badge = `<span class="badge">cached</span>`;
  return [
    `<button type="button" class="mopt" data-model="${id}" aria-pressed="${active}">`,
    `<span class="mo-t"><b>${esc(note.name)}</b>${badge}</span>`,
    `<span class="mo-d">${esc(note.short)}</span>`,
    `<span class="mo-d">${esc(s.costLine(id))}</span>`,
    `</button>`,
  ].join("");
}

const BACKEND_CHOICES = [
  ["auto", "Auto"],
  ["webgpu", "WebGPU"],
  ["wasm", "CPU"],
];

function backendSwitch(s) {
  const buttons = BACKEND_CHOICES.map(([value, label]) => {
    const unavailable = value === "webgpu" && !navigator.gpu;
    const disabled = unavailable ? ` disabled title="WebGPU is not available in this browser"` : "";
    return `<button type="button" data-be="${value}" aria-pressed="${s.backend === value}"${disabled}>${label}</button>`;
  });
  return `<div class="seg" role="group" aria-label="Backend">${buttons.join("")}</div>`;
}

/** The bottom of the panel: progress while loading, the backend once loaded, else a load button. */
function panelAction(s) {
  if (s.status === "loading") {
    return [
      `<div class="mp-load">`,
      `<div class="mp-row"><span class="spin"></span><span class="mp-label" data-f="label"></span>`,
      `<button type="button" class="btn small ghost" data-act="cancel">Cancel</button></div>`,
      `<div class="progress"><i></i></div>`,
      `</div>`,
    ].join("");
  }
  if (s.status === "ready") {
    const { info } = s;
    const kind = info.backend === "webgpu" ? "gpu" : "cpu";
    const detail = `${esc(info.gpu || "")} · ready in ${fmtMs(info.loadMs)}${info.pack?.cached ? " from cache" : ""}`;
    return [
      `<div class="mp-ready">`,
      `<span class="badge ${kind}"><span class="dot"></span>${esc(backendLabel(info))}</span>`,
      `<span class="tiny faint">${detail}</span>`,
      `<span class="spacer"></span>`,
      `<button type="button" class="btn small ghost" data-act="unload">Unload</button>`,
      `</div>`,
      s.storageError ? `<p class="tiny gate-err">Not stored for next time (${esc(s.storageError)}). Free some disk space or clear stored packs, and the next load will keep it.</p>` : "",
    ].join("");
  }
  const failed = s.status === "error";
  return [
    failed ? `<p class="gate-err">Could not load: ${esc(s.error)}</p>` : "",
    `<button type="button" class="btn primary wide" data-act="load">${failed ? "Retry" : `Load ${esc(s.nameOf())}`}</button>`,
  ].join("");
}

let panelKey = "";
function renderPanel(s) {
  // rebuild only when what the panel shows changes; progress updates below touch the bar alone
  const key = `${s.status}|${s.model}|${s.backend}|${JSON.stringify(s.cached)}|${s.error}|${s.storageError}|${s.storage?.bytes}`;
  if (key !== panelKey) {
    panelKey = key;
    const options = Object.keys(MODEL_NOTES).map((id) => modelOption(s, id));
    const stored = s.storage?.available ? `Stored packs: ${fmtBytes(s.storage.bytes)}` : "No persistent storage";
    panel.innerHTML = `<div class="mp-h">Model <span class="tiny faint">shared by every page, kept for this session</span></div>
      <div class="mp-models">${options.join("")}</div>
      <div class="mp-be"><span class="tiny muted">Backend</span>${backendSwitch(s)}</div>
      ${panelAction(s)}
      <div class="mp-foot">
        <span class="tiny faint">${stored}</span>
        <button type="button" class="linkbtn" data-act="clear">Clear stored packs</button>
      </div>`;
  }
  if (s.status === "loading") {
    panel.querySelector('[data-f="label"]').textContent = s.progress?.label || "";
    const bar = panel.querySelector(".mp-load .progress");
    bar.classList.toggle("indet", !!s.progress?.indet);
    bar.firstElementChild.style.width = `${Math.round((s.progress?.frac || 0) * 100)}%`;
  }
}

function openPanel(open) {
  panel.classList.toggle("hidden", !open);
  chip.setAttribute("aria-expanded", String(open));
  if (open) renderPanel(session);
}
chip.addEventListener("click", () => openPanel(panel.classList.contains("hidden")));
document.addEventListener("pointerdown", (e) => {
  if (!panel.classList.contains("hidden") && !e.target.closest(".mm")) openPanel(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !panel.classList.contains("hidden")) openPanel(false);
});
panel.addEventListener("click", async (e) => {
  const model = e.target.closest("[data-model]")?.dataset.model;
  if (model) {
    const wasActive = session.status === "ready" || session.status === "loading";
    session.select(model);
    // switching while a model is loaded or loading means "use this one instead"
    if (wasActive || session.cached[model]) session.load();
    return;
  }
  const backend = e.target.closest("[data-be]")?.dataset.be;
  if (backend) return session.setBackend(backend);
  const action = e.target.closest("[data-act]")?.dataset.act;
  if (action === "load") session.load();
  if (action === "cancel") session.cancel();
  if (action === "unload") session.unload();
  if (action === "clear") {
    if (session.status === "loading") session.cancel();
    await session.clearCache();
  }
});

session.on((s) => {
  renderChip(s);
  if (!panel.classList.contains("hidden")) renderPanel(s);
});
renderChip(session);

// Router: #/route or #/route/anchor

const main = document.getElementById("app");
const views = new Map(); // route id -> { el, view }
let current = null;

function parseHash() {
  const [id, anchor] = location.hash.replace(/^#\/?/, "").split("/");
  return { route: ROUTES.find((r) => r.id === id) || ROUTES[0], anchor };
}

function viewErrorHTML(error) {
  return `<div class="wrap"><div class="card pad view-error">
    <h2>This page did not load</h2>
    <p class="muted">Part of the site could not be fetched. This usually means the connection dropped or the site
      was updated a moment ago. The model and anything already loaded are unaffected.</p>
    <div class="row">
      <button type="button" class="btn primary" data-retry>Try again</button>
      <a class="btn ghost" href="#/">Go home</a>
    </div>
    <details><summary class="tiny faint">Details</summary><pre class="tiny faint">${esc(error.message)}</pre></details>
  </div></div>`;
}

/** Creates the view's section and mounts its module; a failed import shows a retry card. */
async function mountView(route) {
  const el = document.createElement("section");
  el.hidden = true;
  el.className = `view view-${route.id}`;
  el.dataset.view = route.id;
  main.appendChild(el);
  const entry = { el, view: null };
  views.set(route.id, entry);
  try {
    const viewModule = await route.load();
    entry.view = viewModule.mount(el, { session, navigate }) || {};
  } catch (e) {
    // a stale cache or a dropped connection: offer a retry instead of a raw error
    console.error(e);
    el.innerHTML = viewErrorHTML(e);
    el.querySelector("[data-retry]").addEventListener("click", () => {
      views.delete(route.id);
      el.remove();
      show();
    });
  }
  return entry;
}

let showCount = 0;
async function show() {
  const { route, anchor } = parseHash();
  const ticket = ++showCount;
  document.title = route.title;
  for (const link of header.querySelectorAll("[data-route]")) {
    link.toggleAttribute("aria-current", link.dataset.route === route.id);
  }
  const entry = views.get(route.id) ?? (await mountView(route));
  if (ticket !== showCount) return; // another route was picked while this one loaded
  if (current && current !== route.id) {
    const previous = views.get(current);
    previous.el.hidden = true;
    previous.view?.hide?.();
  }
  current = route.id;
  entry.el.hidden = false;
  entry.view?.show?.();
  if (anchor) {
    const target = () => entry.el.querySelector(`#${CSS.escape(anchor)}`);
    requestAnimationFrame(() => target()?.scrollIntoView({ behavior: "smooth", block: "start" }));
  } else scrollTo({ top: 0 });
}

export function navigate(path) {
  location.hash = `#/${path}`;
}

addEventListener("hashchange", show);
show();
session.boot();

// read-only handle for tests
window.kevala_site = { session, views };
