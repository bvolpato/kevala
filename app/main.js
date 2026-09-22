// The site shell: header tabs, the model menu, and a hash router that keeps every visited view
// alive. Views are ES modules loaded on first visit; hiding one calls its hide() so games pause
// and timers stop, and the model keeps loading in the session whatever the tab.

import { session, MODELS, MODEL_NOTES, LOCAL, SHOT } from "./session.js";
import { esc, fmtBytes, fmtMs, backendLabel, REPO } from "./ui.js";

const ROUTES = [
  { id: "home", title: "kevala · decision models in the browser", load: () => import("./views/home.js"), nav: false },
  { id: "playground", label: "Playground", title: "Playground · kevala", load: () => import("./views/playground.js") },
  { id: "tetris", label: "Tetris", title: "Tetris played by a decision model · kevala", load: () => import("./views/tetris.js") },
  { id: "guardrail", label: "Guardrail", title: "Prompt guardrail · kevala", load: () => import("./views/guardrail.js") },
  { id: "inbox", label: "Inbox", title: "Inbox triage · kevala", load: () => import("./views/inbox.js") },
  { id: "how", label: "How it works", title: "How kevala works", load: () => import("./views/how.js") },
];

const LOGO = `<svg viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="wg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7aa2ff"/><stop offset=".55" stop-color="#45e0c0"/><stop offset="1" stop-color="#c592ff"/></linearGradient></defs><rect x="1" y="1" width="30" height="30" rx="9" fill="#0f131b" stroke="url(#wg)" stroke-width="2"/><path d="M7 11l4 11 5-8 5 8 4-11" fill="none" stroke="url(#wg)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

// ---------------------------------------------------------------------------------------------
// header

const header = document.querySelector("header.nav");
header.innerHTML = `<div class="wrap">
  <a class="brand" href="#/">${LOGO}<span>kevala</span></a>
  <nav class="tabs-nav" aria-label="Sections">${ROUTES.filter((r) => r.nav !== false)
    .map((r) => `<a href="#/${r.id}" data-route="${r.id}">${esc(r.label)}</a>`)
    .join("")}</nav>
  <span class="spacer"></span>
  ${LOCAL && !SHOT ? `<span class="badge warn" title="Loading packs from this server's tmp/">dev packs</span>` : ""}
  <div class="mm">
    <button type="button" class="mchip" aria-haspopup="dialog" aria-expanded="false"><span class="mc-dot"></span><span class="mc-name"></span><span class="mc-state"></span><svg class="mc-caret" viewBox="0 0 10 6" aria-hidden="true"><path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button>
    <div class="mpanel hidden" role="dialog" aria-label="Model"></div>
  </div>
  <a class="nav-gh" href="${REPO}" title="Source on GitHub">GitHub</a>
</div><div class="gprog hidden"><i></i></div>`;

const chip = header.querySelector(".mchip");
const panel = header.querySelector(".mpanel");
const gprog = header.querySelector(".gprog");

function renderChip(s) {
  chip.dataset.state = s.status;
  chip.querySelector(".mc-name").textContent = s.nameOf();
  let st = "";
  if (s.status === "ready") st = backendLabel(s.info);
  else if (s.status === "loading") st = s.progress && !s.progress.indet ? `${Math.floor(s.progress.frac * 100)}%` : "loading";
  else if (s.status === "error") st = "error";
  else st = "load";
  chip.querySelector(".mc-state").textContent = st;
  gprog.classList.toggle("hidden", s.status !== "loading");
  if (s.status === "loading") {
    gprog.classList.toggle("indet", !!s.progress?.indet);
    gprog.firstElementChild.style.width = `${Math.round((s.progress?.frac || 0) * 100)}%`;
  }
}

let panelMode = "";
function renderPanel(s) {
  const mode = `${s.status}|${s.model}|${s.backend}|${JSON.stringify(s.cached)}|${s.error}|${s.storage?.bytes}`;
  if (mode !== panelMode) {
    panelMode = mode;
    const gpuOk = !!navigator.gpu;
    const cards = Object.keys(MODEL_NOTES)
      .map((m) => {
        const n = MODEL_NOTES[m];
        const spec = MODELS[m];
        const active = s.model === m;
        const badge = active && s.status === "ready" ? `<span class="badge good">loaded</span>` : s.cached[m] ? `<span class="badge">cached</span>` : `<span class="badge faint-b">${fmtBytes(spec.download)} download</span>`;
        return `<button type="button" class="mopt" data-model="${m}" aria-pressed="${active}"><span class="mo-t"><b>${esc(n.name)}</b>${badge}</span><span class="mo-d">${esc(n.short)}</span><span class="mo-d">${esc(s.costLine(m))}</span></button>`;
      })
      .join("");
    let action = "";
    if (s.status === "loading") {
      action = `<div class="mp-load"><div class="mp-row"><span class="spin"></span><span class="mp-label" data-f="label"></span><button type="button" class="btn small ghost" data-act="cancel">Cancel</button></div><div class="progress"><i></i></div></div>`;
    } else if (s.status === "ready") {
      const i = s.info;
      action = `<div class="mp-ready"><span class="badge ${i.backend === "webgpu" ? "gpu" : "cpu"}"><span class="dot"></span>${esc(backendLabel(i))}</span><span class="tiny faint">${esc(i.gpu || "")} · ready in ${fmtMs(i.loadMs)}${i.pack?.cached ? " from cache" : ""}</span><span class="spacer"></span><button type="button" class="btn small ghost" data-act="unload">Unload</button></div>`;
    } else {
      action = `${s.status === "error" ? `<p class="gate-err">Could not load: ${esc(s.error)}</p>` : ""}<button type="button" class="btn primary wide" data-act="load">${s.status === "error" ? "Retry" : `Load ${esc(s.nameOf())}`}</button>`;
    }
    panel.innerHTML = `<div class="mp-h">Model <span class="tiny faint">one model for every tab, kept for this session</span></div>
      <div class="mp-models">${cards}</div>
      <div class="mp-be"><span class="tiny muted">Backend</span><div class="seg" role="group" aria-label="Backend">${[
        ["auto", "Auto"],
        ["webgpu", "WebGPU"],
        ["wasm", "CPU"],
      ]
        .map(([v, l]) => `<button type="button" data-be="${v}" aria-pressed="${s.backend === v}" ${v === "webgpu" && !gpuOk ? "disabled title='WebGPU is not available in this browser'" : ""}>${l}</button>`)
        .join("")}</div></div>
      ${action}
      <div class="mp-foot"><span class="tiny faint">${s.storage?.available ? `Stored packs: ${fmtBytes(s.storage.bytes)}` : "No persistent storage"}</span><button type="button" class="linkbtn" data-act="clear">Clear stored packs</button></div>`;
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
  const m = e.target.closest("[data-model]");
  if (m) {
    const wasActive = session.status === "ready" || session.status === "loading";
    session.select(m.dataset.model);
    // switching while a model is loaded or loading means "use this one instead"
    if (wasActive || session.cached[m.dataset.model]) session.load();
    return;
  }
  const be = e.target.closest("[data-be]");
  if (be) return session.setBackend(be.dataset.be);
  const a = e.target.closest("[data-act]")?.dataset.act;
  if (a === "load") session.load();
  if (a === "cancel") session.cancel();
  if (a === "unload") session.unload();
  if (a === "clear") {
    if (session.status === "loading") session.cancel();
    await session.clearCache();
  }
});

session.on((s) => {
  renderChip(s);
  if (!panel.classList.contains("hidden")) renderPanel(s);
});
renderChip(session);

// ---------------------------------------------------------------------------------------------
// router: #/route or #/route/anchor

const main = document.getElementById("app");
const views = new Map(); // id -> { el, view, ready }
let current = null;

function parse() {
  const h = location.hash.replace(/^#\/?/, "");
  const [id, anchor] = h.split("/");
  const route = ROUTES.find((r) => r.id === id) || ROUTES[0];
  return { route, anchor };
}

let seq = 0;
async function show() {
  const { route, anchor } = parse();
  const my = ++seq;
  document.title = route.title;
  for (const a of header.querySelectorAll("[data-route]")) a.toggleAttribute("aria-current", a.dataset.route === route.id);
  let v = views.get(route.id);
  if (!v) {
    const el = document.createElement("section");
    el.hidden = true;
    el.className = `view view-${route.id}`;
    el.dataset.view = route.id;
    main.appendChild(el);
    v = { el, view: null };
    views.set(route.id, v);
    try {
      const mod = await route.load();
      v.view = mod.mount(el, { session, navigate }) || {};
    } catch (e) {
      // a stale cache or a dropped connection: offer a retry instead of a raw error
      console.error(e);
      el.innerHTML = `<div class="wrap"><div class="card pad view-error"><h2>This page did not load</h2><p class="muted">Part of the site could not be fetched, usually because the connection dropped or the site was just updated. The model and anything already loaded are fine.</p><div class="row"><button type="button" class="btn primary" data-retry>Try again</button><a class="btn ghost" href="#/">Go home</a></div><details><summary class="tiny faint">Details</summary><pre class="tiny faint">${esc(e.message)}</pre></details></div></div>`;
      el.querySelector("[data-retry]").addEventListener("click", () => {
        views.delete(route.id);
        el.remove();
        show();
      });
    }
  }
  if (my !== seq) return; // another route was picked while this one loaded
  if (current && current !== route.id) {
    const prev = views.get(current);
    prev.el.hidden = true;
    prev.view?.hide?.();
  }
  current = route.id;
  v.el.hidden = false;
  v.view?.show?.();
  if (anchor) requestAnimationFrame(() => v.el.querySelector(`#${CSS.escape(anchor)}`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  else scrollTo({ top: 0 });
}

export function navigate(path) {
  location.hash = `#/${path}`;
}

addEventListener("hashchange", show);
show();
session.boot();

// read-only handle for tests
window.kevala_site = { session, views };
