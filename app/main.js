// The site shell: header tabs, the model menu, and a hash router that keeps every visited view
// alive. Views are ES modules loaded on first visit; hiding one calls its hide() so games pause
// and timers stop, and the model keeps loading in the session whatever the tab.

import { session, MODELS, MODEL_NOTES, LOCAL, SHOT, FROM, requiresWebGPU } from "./session.js";
import { esc, fmtBytes, fmtMs, backendLabel, cpuReason, logo, REPO } from "./ui.js";

/** A view and its tab in the header. */
const defineRoute = (id, label, title) => ({ id, label, title, load: () => import(`./views/${id}.js`) });

const ROUTES = [
  defineRoute("home", "Home", "kevala · decision models that run in the page"),
  defineRoute("playground", "Playground", "Playground · kevala"),
  defineRoute("reaction", "Reaction Recorder", "Reaction Recorder · kevala"),
  defineRoute("tetris", "Tetris", "Model-guided Tetris · kevala"),
  defineRoute("guardrail", "Injection gate", "Prompt injection gate · kevala"),
  defineRoute("inbox", "Inbox", "Inbox triage · kevala"),
  defineRoute("how", "How it works", "How kevala works"),
];

const LOGO = logo("wg");

const CARET =
  `<svg class="mc-caret" viewBox="0 0 10 6" aria-hidden="true">` +
  `<path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;

// Header

const header = document.querySelector("header.nav");
const navLinks = ROUTES
  .map((r) => `<a href="#/${r.id === "home" ? "" : r.id}" data-route="${r.id}">${esc(r.label)}</a>`)
  .join("");
const devBadge = LOCAL && !SHOT ? `<span class="badge good" title="Converted model packs are served by this preview">local packs</span>` : "";
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

const MODEL_FAMILIES = {
  kev: {
    name: "Kev",
    short: "Trained decision head · v1.0",
    models: [["kev-0.8b", "0.8B"], ["kev-4b", "4B"], ["kev-9b", "9B"]],
  },
  h2o: {
    name: "H2O-Lightning",
    short: "Calibrated decisions · 4B",
    models: [["h2o-lightning-4b", "4B"]],
  },
  clef: {
    name: "Cloudflare Clef-flash",
    short: "Qwen3.5 with Cloudflare's joint decision head",
    models: [["clef-flash", "Flash"]],
  },
  semif: {
    name: "SemIf",
    short: "Direct option scoring",
    models: [["semif-qwen3.5-0.8b", "0.8B"], ["semif-qwen3.5-2b", "2B"], ["semif-qwen3.5-4b", "4B"]],
  },
  gemma4: {
    name: "Gemma 4",
    short: "Images · E2B · direct option scoring",
    models: [["gemma-4-e2b", "E2B"]],
  },
};

const FAMILY_BY_MODEL = Object.fromEntries(
  Object.entries(MODEL_FAMILIES).flatMap(([family, spec]) => spec.models.map(([id]) => [id, family])),
);

function familyModels(family) {
  return MODEL_FAMILIES[family].models.filter(([id]) => MODELS[id] && (LOCAL || (FROM === "checkpoint"
    ? MODELS[id].browserConvert
    : MODELS[id].browserConvert || MODELS[id].hosted || MODELS[id].nativeConversion)));
}

function modelPackBytes(id) {
  const spec = MODELS[id];
  const precision = session.precisionFor(id);
  if (!spec) return 0;
  if (session.customPackUrl(id)) return 0;
  if (session.localPackFor(id)) return session.localPackFor(id).bytes;
  if (precision === "q8" && FROM === "pack" && spec.hosted && spec.pack) return spec.pack;
  return spec.precisionPackBytes?.[precision] || spec.download || spec.pack;
}

function modelSizeLabel(id) {
  const spec = MODELS[id];
  const precision = session.precisionFor(id);
  if (session.customPackUrl(id)) return "Custom pack";
  if (session.localOnly && !session.localPackFor(id)) return "Local pack not found";
  if (FROM === "checkpoint" && spec?.browserConvert) return `${spec.downloadApprox ? "about " : ""}${fmtBytes(spec.download)} source checkpoint`;
  const bytes = modelPackBytes(id);
  if (!bytes) return "Pack size unavailable";
  if (session.localPackFor(id)) return `${fmtBytes(bytes)} · ${precision.toUpperCase()}`;
  if (precision === "q8" && spec?.hosted && FROM === "pack" && spec.pack) return `${fmtBytes(bytes)} · Q8`;
  if (spec?.precisionPackBytes?.[precision]) {
    const approx = spec.precisionPackApprox?.[precision] !== false;
    return `${approx ? "about " : ""}${fmtBytes(bytes)} · ${precision.toUpperCase()}`;
  }
  return `${fmtBytes(bytes)} source checkpoint`;
}

function modelBadge(s, id) {
  if (s.model === id && s.status === "ready") return `<span class="badge good">loaded</span>`;
  if (s.cached[id]) return `<span class="badge">cached</span>`;
  if (s.customPackUrl(id)) return `<span class="badge faint-b">custom pack</span>`;
  if (s.localPackFor(id)) return `<span class="badge good">ready</span>`;
  if (s.localOnly) return `<span class="badge faint-b">not found locally</span>`;
  const nativePackMissing = MODELS[id]?.nativeConversion
    && !(FROM === "pack" && session.precisionFor(id) === "q8" && MODELS[id]?.hosted);
  if (nativePackMissing) return `<span class="badge faint-b">pack URL required</span>`;
  const bytes = modelPackBytes(id);
  if (!bytes) return "";
  const spec = MODELS[id];
  const size = !spec.hosted && spec.browserConvert ? `${spec.downloadApprox ? "about " : ""}${fmtBytes(spec.download)} source` : `${fmtBytes(bytes)} download`;
  return `<span class="badge faint-b">${size}</span>`;
}

function modelStateLine(s, id) {
  if (s.model === id && s.status === "ready" && s.info) {
    return `${backendLabel(s.info)} · loaded`;
  }
  if (s.customPackUrl(id)) return "Precision declared by your pack";
  if (s.localPackFor(id)) return requiresWebGPU(id, s.precisionFor(id)) ? "WebGPU required" : "WebGPU or CPU";
  if (s.localOnly) return "No matching local pack found";
  if (s.cached[id]) return "Stored in this browser";
  if (FROM === "checkpoint") return "Converts in the browser";
  if (MODELS[id]?.nativeConversion) return "Pack URL required";
  return MODELS[id]?.hosted ? "Downloads on first load" : "Converts on first load";
}

function familyValueText(s, id) {
  const note = MODEL_NOTES[id];
  const state = s.model === id && s.status === "ready" ? "loaded" : s.localPackFor(id) ? "ready on this server" : s.localOnly ? "local pack missing" : s.cached[id] ? "cached" : MODELS[id]?.nativeConversion ? "native pack required" : "available";
  const precision = session.customPackUrl(id) ? "fixed pack precision" : `${session.precisionFor(id)} precision`;
  return `${note?.name || id}; ${modelSizeLabel(id)}; ${state}; ${precision}`;
}

function visionControls(s, id) {
  const spec = MODELS[id];
  if (!spec?.modalities?.includes("image") || !spec.vision) return "";
  const local = s.localVisionPackFor(id);
  const enabled = s.visionEnabledFor(id);
  const url = !local && s.visionUrlModel === id ? s.visionUrl : "";
  const decoderBytes = spec.pack ? `${fmtBytes(spec.pack)} Q8 decoder` : "Q8 decoder";
  const visionBytes = spec.vision.pack ? fmtBytes(spec.vision.pack) : "about 337 MB";
  return [
    `<div class="mf-control" data-vision-controls="${id}">`,
    `<label class="tiny muted"><input type="checkbox" data-vision-enabled${enabled ? " checked" : ""}> Enable images</label>`,
    `<span class="tiny mf-info" data-vision-status></span>`,
    local ? "" : `<label class="mp-pack-url">BF16 vision-pack URL (required for images)<input type="url" data-vision-url aria-label="BF16 vision-pack URL" value="${esc(url)}" spellcheck="false" autocomplete="url"><span class="tiny faint">Nothing downloads until you press Load.</span></label>`,
    `<span class="tiny mf-info">${esc(decoderBytes)} + ${esc(visionBytes)} BF16 vision pack. Images require WebGPU and about 2.9 GB of GPU weights. Browser RAM reached 6 GB in local testing; leave extra working space.</span>`,
    `</div>`,
  ].join("");
}

function modelOption(s, id) {
  const note = MODEL_NOTES[id];
  const spec = MODELS[id];
  if (!note || !spec) return "";
  const active = s.model === id;
  return [
    `<button type="button" class="mopt" data-model="${id}" aria-pressed="${active}">`,
    `<span class="mo-t"><b>${esc(note.name)}</b><span data-model-status>${modelBadge(s, id)}</span></span>`,
    `<span class="mo-d">Fast encoder · 421M</span>`,
    `<span class="mo-d mo-meta" data-model-cost>${esc(modelSizeLabel(id))} · ${esc(modelStateLine(s, id))}</span>`,
    `</button>`,
  ].join("");
}

function familyOption(s, family) {
  const spec = MODEL_FAMILIES[family];
  const models = familyModels(family);
  if (!spec || !models.length) return "";
  const active = FAMILY_BY_MODEL[s.model] === family;
  const selected = active && models.some(([id]) => id === s.model) ? s.model : models[0][0];
  const selectedIndex = Math.max(0, models.findIndex(([id]) => id === selected));
  const infoId = `mf-info-${family}`;
  const ticks = models.map(([, label]) => `<span>${esc(label)}</span>`).join("");
  return [
    `<div class="mfamily" data-family-card="${family}">`,
    `<button type="button" class="mopt mf-select" data-family="${family}" aria-pressed="${active}">`,
    `<span class="mo-t"><b>${esc(spec.name)}</b><span data-family-status>${modelBadge(s, selected)}</span></span>`,
    `<span class="mo-d">${esc(spec.short)}</span>`,
    `</button>`,
    `<div class="mf-control" role="group" aria-label="${esc(spec.name)} settings">`,
    models.length > 1 ? `<div class="mf-size-line"><span class="tiny muted" data-family-size-label>${esc(spec.name)} size</span><b data-family-size>${esc(models[selectedIndex][1])}</b></div>` : "",
    models.length > 1 ? `<input type="range" class="mf-slider" data-family-slider="${family}" min="0" max="${models.length - 1}" step="1" value="${selectedIndex}" aria-label="${esc(spec.name)} model size" aria-valuetext="${esc(familyValueText(s, selected))}" aria-describedby="${infoId}">` : "",
    models.length > 1 ? `<div class="mf-ticks" aria-hidden="true">${ticks}</div>` : "",
    `<div class="mf-meta"><span class="tiny mf-pack" data-family-pack>${esc(modelSizeLabel(selected))}</span><span class="tiny mf-info" id="${infoId}" data-family-info>${esc(modelStateLine(s, selected))}</span></div>`,
    visionControls(s, selected),
    `</div>`,
    `</div>`,
  ].join("");
}

function modelChoices(s) {
  // Entries without a published pack remain visible when they have a pinned native conversion path.
  const known = Object.keys(MODEL_NOTES).filter((id) => MODELS[id] && (MODELS[id].hosted || MODELS[id].browserConvert || MODELS[id].nativeConversion || LOCAL));
  const renderLaya = known.includes("laya") ? modelOption(s, "laya") : "";
  const renderFamily = (family) => familyModels(family).some(([id]) => known.includes(id)) ? familyOption(s, family) : "";
  const models = [renderLaya, ...Object.keys(MODEL_FAMILIES).map(renderFamily)].filter(Boolean).join("");
  return `<div class="mp-models">${models}</div>`;
}

const BACKEND_CHOICES = [
  ["auto", "Auto"],
  ["webgpu", "WebGPU"],
  ["wasm", "CPU"],
];

function backendUnavailable(s, backend) {
  if (backend === "webgpu" && (!navigator.gpu || s.gpuAvailable === false)) return "WebGPU is not available in this browser";
  if (backend === "wasm" && requiresWebGPU(s.model, s.precisionFor())) return "This model and precision exceed the WebAssembly pack limit";
  return "";
}

function precisionSelector(s) {
  const choices = s.precisionChoices();
  if (choices.length < 2) return "";
  const value = s.precisionFor();
  const options = choices.map((precision) => {
    const label = precision === "bf16" ? "BF16 · default" : precision === "q8" ? "Q8 · quantized, smaller" : `${precision.toUpperCase()} · full precision`;
    return `<option value="${precision}"${precision === value ? " selected" : ""}>${label}</option>`;
  });
  return `<label class="mp-precision">Precision<select data-precision aria-label="Pack precision">${options.join("")}</select><span class="tiny faint" data-precision-note>BF16 is unquantized. Q8 saves memory; probabilities may change.</span></label>`;
}

function nativePackUrlControl() {
  return `<div class="mp-pack-override" data-native-pack-url-wrap hidden><label class="mp-pack-url">Converted .kevala pack URL<input type="url" data-native-pack-url aria-label="Converted pack URL (required)" value="" spellcheck="false" autocomplete="url"><span class="tiny faint">No pack is available for this precision. Paste the URL of a converted pack.</span></label></div>`;
}

function backendSwitch(s) {
  const buttons = BACKEND_CHOICES.map(([value, label]) => {
    const unavailable = backendUnavailable(s, value);
    const disabled = unavailable ? ` disabled title="${esc(unavailable)}"` : "";
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
      cpuReason(info) ? `<p class="tiny mp-note">${esc(cpuReason(info))}</p>` : "",
      s.storageError ? `<p class="tiny gate-err">Not stored for next time (${esc(s.storageError)}). Free some disk space or clear stored packs, and the next load will keep it.</p>` : "",
    ].join("");
  }
  const failed = s.status === "error";
  const unavailable = s.loadUnavailable();
  const failureLabel = s.errorKind === "runtime" ? "Model unavailable" : "Could not load";
  return [
    failed ? `<p class="gate-err">${failureLabel}: ${esc(s.error)}</p>` : "",
    unavailable && !failed ? `<p class="tiny mp-note">${esc(unavailable)}</p>` : "",
    `<button type="button" class="btn primary wide" data-act="load"${unavailable ? ` disabled title="${esc(unavailable)}"` : ""}>${failed ? "Retry" : `Load ${esc(s.nameOf())}`}</button>`,
  ].join("");
}

let panelKey = "";
let panelActionKey = "";
let sliderPointerActive = false;
let panelRenderPending = false;

function panelFocus() {
  const active = document.activeElement;
  if (active?.matches("[data-family-slider]")) return { slider: active.dataset.familySlider };
  if (active?.matches("[data-model], [data-family]")) {
    return { model: active.dataset.model, family: active.dataset.family };
  }
  return null;
}

function restorePanelFocus(focus) {
  if (!focus) return;
  let target = null;
  if (focus.slider) target = panel.querySelector(`[data-family-slider="${CSS.escape(focus.slider)}"]`);
  else if (focus.family) target = panel.querySelector(`[data-family="${CSS.escape(focus.family)}"]`);
  else if (focus.model) target = panel.querySelector(`[data-model="${CSS.escape(focus.model)}"]`);
  target?.focus({ preventScroll: true });
}

function syncPanelSelection(s) {
  for (const button of panel.querySelectorAll("[data-be]")) {
    const reason = backendUnavailable(s, button.dataset.be);
    button.disabled = !!reason;
    button.title = reason;
  }
  for (const option of panel.querySelectorAll("[data-model]")) {
    const id = option.dataset.model;
    option.setAttribute("aria-pressed", String(s.model === id));
    const status = option.querySelector("[data-model-status]");
    const cost = option.querySelector("[data-model-cost]");
    if (status) status.innerHTML = modelBadge(s, id);
    if (cost) cost.textContent = `${modelSizeLabel(id)} · ${modelStateLine(s, id)}`;
  }
  for (const card of panel.querySelectorAll("[data-family-card]")) {
    const family = card.dataset.familyCard;
    const spec = MODEL_FAMILIES[family];
    const models = familyModels(family);
    if (!spec || !models.length) continue;
    const active = FAMILY_BY_MODEL[s.model] === family;
    const selected = active && models.some(([id]) => id === s.model) ? s.model : models[0][0];
    const selectedIndex = Math.max(0, models.findIndex(([id]) => id === selected));
    const select = card.querySelector("[data-family]");
    const slider = card.querySelector("[data-family-slider]");
    if (select) select.setAttribute("aria-pressed", String(active));
    card.classList.toggle("active", active);
    // Keep the native range element and its focus during a model selection. This also keeps a
    // pointer drag alive while session.select() emits its unload/selection changes.
    if (slider && document.activeElement !== slider && !sliderPointerActive) slider.value = String(selectedIndex);
    if (slider) slider.setAttribute("aria-valuetext", familyValueText(s, selected));
    const status = card.querySelector("[data-family-status]");
    const size = card.querySelector("[data-family-size]");
    const pack = card.querySelector("[data-family-pack]");
    const info = card.querySelector("[data-family-info]");
    if (status) status.innerHTML = modelBadge(s, selected);
    if (size) size.textContent = models[selectedIndex][1];
    if (pack) pack.textContent = modelSizeLabel(selected);
    if (info) info.textContent = modelStateLine(s, selected);
    const vision = card.querySelector("[data-vision-controls]");
    if (vision) {
      const enabled = vision.querySelector("[data-vision-enabled]");
      if (enabled) enabled.checked = s.visionEnabledFor(selected);
      const status = vision.querySelector("[data-vision-status]");
      const localVision = s.localVisionPackFor(selected);
      if (status) status.textContent = localVision
        ? `Vision pack ready on this server · BF16 · ${fmtBytes(localVision.bytes)}`
        : s.visionUrlFor(selected)
          ? "Vision-pack URL supplied. The pack is checked when you press Load."
          : "No local vision pack is available. Text mode still works. Paste a BF16 vision-pack URL to enable images.";
      const url = vision.querySelector("[data-vision-url]");
      if (url) {
        const value = s.visionUrlModel === selected ? s.visionUrl : "";
        if (url.value !== value) url.value = value;
        url.setAttribute("aria-required", String(s.visionEnabledFor(selected)));
      }
    }
  }
  const loadButton = panel.querySelector('[data-act="load"]');
  if (loadButton && s.status !== "error") loadButton.textContent = `Load ${s.nameOf()}`;
  if (loadButton) {
    loadButton.disabled = !!s.loadUnavailable();
    loadButton.title = s.loadUnavailable();
  }
  const precision = panel.querySelector("[data-precision]");
  if (precision) {
    const fixedPack = !!s.customPackUrl();
    if (fixedPack && !precision.disabled) {
      precision.dataset.formatOptions = precision.innerHTML;
      precision.innerHTML = '<option value="">Fixed by pack</option>';
    } else if (!fixedPack && precision.disabled && precision.dataset.formatOptions) {
      precision.innerHTML = precision.dataset.formatOptions;
    }
    precision.disabled = fixedPack;
    precision.value = fixedPack ? "" : s.precisionFor();
    precision.title = precision.disabled ? "The supplied .kevala pack has a fixed precision." : "";
    const note = panel.querySelector("[data-precision-note]");
    if (note) note.textContent = precision.disabled
      ? "The supplied pack declares its precision."
      : "BF16 is unquantized. Q8 saves memory; probabilities may change.";
  }
  const packUrl = panel.querySelector("[data-native-pack-url]");
  const packUrlWrap = panel.querySelector("[data-native-pack-url-wrap]");
  if (packUrl && packUrlWrap) {
    packUrlWrap.hidden = !MODELS[s.model]?.nativeConversion || s.localOnly || !!s.localPackFor(s.model);
    const value = s.customPackUrl();
    if (packUrl.value !== value) packUrl.value = value;
  }
}

function renderPanel(s) {
  // Keep the model controls mounted across load, error, backend, and storage changes. In
  // particular, replacing a focused range input after each arrow press makes browser automation
  // and keyboard users lose the control even when focus is restored to its replacement.
  const key = FAMILY_BY_MODEL[s.model] || s.model;
  let focus = null;
  if (key !== panelKey) {
    if (sliderPointerActive && panel.childElementCount) {
      panelRenderPending = true;
      syncPanelSelection(s);
      return;
    }
    focus = panelFocus();
    panelKey = key;
    const stored = s.storage?.available ? `Stored packs: ${fmtBytes(s.storage.bytes)}` : "No persistent storage";
    panel.innerHTML = `<div class="mp-h">Choose a model</div>
      <p class="mp-availability tiny" data-local-availability>Up to 4B · Gemma E2B image pilot · Load on demand${LOCAL ? " · Local packs are marked below" : ""}</p>
      ${modelChoices(s)}
      ${precisionSelector(s)}
      ${nativePackUrlControl()}
      <div data-panel-action></div>
      <details class="mp-advanced" data-advanced><summary class="tiny muted">Advanced</summary><div class="mp-advanced-controls">
        <div class="mp-be"><span class="tiny muted">Backend</span>${backendSwitch(s)}</div>
        <div class="mp-foot">
          <span class="tiny faint" data-storage-label>${stored}</span>
          <button type="button" class="linkbtn" data-act="clear">Clear stored packs</button>
        </div>
      </div></details>`;
    panelActionKey = "";
  }
  const actionKey = [s.status, s.model, s.precisionFor(), s.backend, s.loadUnavailable(), s.error, s.storageError, s.info?.backend, s.info?.gpu, s.info?.loadMs, s.info?.pack?.cached].join("|");
  if (actionKey !== panelActionKey) {
    panel.querySelector("[data-panel-action]").innerHTML = panelAction(s);
    panelActionKey = actionKey;
  }
  const stored = s.storage?.available ? `Stored packs: ${fmtBytes(s.storage.bytes)}` : "No persistent storage";
  panel.querySelector("[data-storage-label]").textContent = stored;
  const precision = panel.querySelector("[data-precision]");
  if (precision) precision.value = s.precisionFor();
  syncPanelSelection(s);
  restorePanelFocus(focus);
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
document.addEventListener("kevala:open-model-picker", () => {
  openPanel(true);
  chip.focus({ preventScroll: true });
});
document.addEventListener("pointerdown", (e) => {
  if (!panel.classList.contains("hidden") && !e.target.closest(".mm")) openPanel(false);
});
panel.addEventListener("pointerdown", (e) => {
  if (e.target.closest("[data-family-slider]")) sliderPointerActive = true;
});
const finishSliderPointer = () => {
  if (!sliderPointerActive) return;
  sliderPointerActive = false;
  if (panelRenderPending) {
    panelRenderPending = false;
    renderPanel(session);
  }
};
document.addEventListener("pointerup", finishSliderPointer);
document.addEventListener("pointercancel", finishSliderPointer);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !panel.classList.contains("hidden")) openPanel(false);
});
panel.addEventListener("input", (e) => {
  const slider = e.target.closest("[data-family-slider]");
  if (!slider) return;
  const models = familyModels(slider.dataset.familySlider);
  const id = models[Number(slider.value)]?.[0];
  if (id && id !== session.model) session.select(id);
});
panel.addEventListener("change", (e) => {
  const precision = e.target.closest("[data-precision]")?.value;
  const visionEnabled = e.target.closest("[data-vision-enabled]");
  if (precision) session.setPrecision(precision);
  if (visionEnabled) session.setVisionEnabled(visionEnabled.checked);
  if (e.target.closest("[data-native-pack-url]")) session.refreshCache();
});
panel.addEventListener("input", (e) => {
  const input = e.target.closest("[data-native-pack-url]");
  if (input) session.setCustomUrl(input.value, session.model);
  const visionUrl = e.target.closest("[data-vision-url]");
  if (visionUrl) session.setVisionUrl(visionUrl.value, session.model);
});
panel.addEventListener("click", async (e) => {
  const model = e.target.closest("[data-model]")?.dataset.model;
  if (model) {
    session.select(model);
    return;
  }
  const family = e.target.closest("[data-family]")?.dataset.family;
  if (family) {
    const models = familyModels(family);
    const defaultModel = models[0]?.[0];
    if (defaultModel && FAMILY_BY_MODEL[session.model] !== family) session.select(defaultModel);
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

// on narrow screens the tabs scroll sideways: fade the right edge while more tabs are hidden there
const tabs = header.querySelector(".tabs-nav");
const markHiddenTabs = () => tabs.classList.toggle("more", tabs.scrollLeft + tabs.clientWidth < tabs.scrollWidth - 2);
tabs.addEventListener("scroll", markHiddenTabs, { passive: true });
addEventListener("resize", markHiddenTabs);
markHiddenTabs();

let showCount = 0;
async function show() {
  const { route, anchor } = parseHash();
  const ticket = ++showCount;
  document.title = route.title;
  for (const link of header.querySelectorAll("[data-route]")) {
    link.toggleAttribute("aria-current", link.dataset.route === route.id);
    if (link.dataset.route === route.id && link.closest(".tabs-nav")) link.scrollIntoView({ block: "nearest", inline: "nearest" });
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
