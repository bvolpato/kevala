import { esc } from "../ui.js";

const REPORT_PATH = "docs/benchmarks/kernel-profile-2026-11-07/summary.json";
const OPTIMIZATION_REPORT_PATH = "docs/benchmarks/model-optimization-2026-11-06/summary.json";
const RAW_REPORT_URL = "https://github.com/bvolpato/kevala/blob/main/docs/benchmarks/kernel-profile-2026-11-07/summary.json";
const RAW_OPTIMIZATION_REPORT_URL = "https://github.com/bvolpato/kevala/blob/main/docs/benchmarks/model-optimization-2026-11-06/summary.json";
const METHOD_URL = "https://github.com/bvolpato/kevala/blob/main/docs/kernel-profiling.md";

const text = (value, fallback = "Not reported") => {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "object") return compact(value);
  return String(value);
};

function compact(value, limit = 220) {
  let result;
  if (Array.isArray(value)) {
    const items = value.slice(0, 8).map((item) => compact(item, 70));
    result = items.join(", ");
    if (value.length > 8) result += `, +${value.length - 8} more`;
  } else if (value && typeof value === "object") {
    result = Object.entries(value).slice(0, 8).map(([key, item]) => `${key}: ${compact(item, 70)}`).join(" · ");
  } else {
    result = String(value);
  }
  return result.length > limit ? `${result.slice(0, limit - 1)}…` : result;
}

function statusClass(value) {
  const status = String(value ?? "").trim().toLowerCase().replaceAll("_", "-");
  if (["", "unknown", "not reported", "unmeasured", "not measured", "not-measured", "incomplete", "pending", "skipped", "blocked", "unsupported", "n/a", "na"].includes(status)) return "quiet";
  if (["fail", "failed", "error", "regressed", "regression"].includes(status)) return "bad";
  if (["pass", "passed", "success", "successful", "measured", "complete", "completed", "keep", "kept", "retain", "retained", "accept", "accepted", "improved", "best"].includes(status)) return "good";
  if (["partial", "warning", "warn", "mixed"].includes(status)) return "warn";
  return "quiet";
}

function badge(value, label = value) {
  const state = text(value, "Not reported");
  return `<span class="profile-badge ${statusClass(state)}" title="${esc(state)}">${esc(label ?? state)}</span>`;
}

function number(value) {
  if (value === null || value === undefined || typeof value === "boolean") return null;
  if (typeof value === "string" && value.trim() === "") return null;
  if (typeof value !== "number" && typeof value !== "string") return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function milliseconds(value) {
  const parsed = number(value);
  if (parsed === null) return "Not measured";
  return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(parsed)} ms`;
}

function kernelTiming(value) {
  const parsed = number(value);
  if (parsed === null) return "Not measured";
  if (parsed > 0 && parsed < 1) {
    return `${new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 }).format(parsed * 1000)} µs`;
  }
  return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(parsed)} ms`;
}

function samples(value) {
  if (Array.isArray(value)) return `${value.length} samples`;
  const count = number(value);
  return count === null ? "sample count not reported" : `${count} samples`;
}

function dateLabel(value) {
  if (!value) return "Not reported";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return text(value);
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
  }).format(date);
}

function browserLabel(browser) {
  if (browser && typeof browser === "object") {
    const name = text(browser.name ?? browser.product, "");
    const version = text(browser.version, "");
    return [name, version].filter(Boolean).join(" ") || compact(browser);
  }
  return text(browser);
}

function safeUrl(value) {
  if (!value) return "";
  try {
    const url = new URL(String(value), document.baseURI);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : "";
  } catch {
    return "";
  }
}

function measuredMetric(metric) {
  return metric && number(metric.medianMs) !== null;
}

function caseLabel(testCase) {
  const names = {
    "1 question, ~40 tokens": "Short fixture · one question",
    "1 question, ~128 tokens": "Medium fixture · one question",
    "1 question, ~512 tokens": "Long fixture · one question",
  };
  const name = names[testCase.name] || text(testCase.name, "Case");
  const observed = Array.isArray(testCase.tokenCounts)
    ? [...new Set(testCase.tokenCounts.filter(Number.isSafeInteger))].sort((a, b) => a - b)
    : [];
  const range = testCase.tokenRange;
  const counts = observed.length
    ? observed.length === 1 ? String(observed[0]) : `${observed.slice(0, -1).join(", ")} or ${observed.at(-1)}`
    : Array.isArray(range) && range.length === 2 && range.every(Number.isSafeInteger)
      ? range[0] === range[1] ? String(range[0]) : `${range[0]}–${range[1]}`
      : testCase.tokens === undefined ? "" : text(testCase.tokens);
  const count = counts ? `${counts} input tokens` : "";
  return `${name}${count ? ` · ${count}` : ""}`;
}

function metricList(model, kind) {
  const cases = Array.isArray(model.cases) ? model.cases : [];
  if (!cases.length) {
    return `<div class="profile-metric-empty"><span class="profile-missing">Not measured</span><small>${esc(text(model.reason, "No case results were recorded."))}</small></div>`;
  }
  return `<ul class="profile-metric-list">${cases.map((testCase) => {
    const metric = testCase[kind];
    const value = metric?.medianMs;
    const hasTiming = measuredMetric(metric);
    const note = hasTiming ? samples(metric.samples) : text(metric?.reason ?? testCase.reason ?? model.reason, "No result was recorded for this case.");
    const timing = hasTiming ? milliseconds(value) : "Not measured";
    const extra = kind === "wall" && number(metric?.p95Ms) !== null ? ` · p95 ${milliseconds(metric.p95Ms)}` : "";
    return `<li><span class="profile-metric-case">${esc(caseLabel(testCase))}</span><b>${esc(timing)}${esc(extra)}</b><small>${esc(note)}</small></li>`;
  }).join("")}</ul>`;
}

function sampleCell(metric) {
  return metric ? samples(metric.samples) : "Not reported";
}

function hasDispatchCounts(kernels) {
  return Array.isArray(kernels) && kernels.some((kernel) => number(kernel.dispatchedSamples) !== null || number(kernel.totalSamples) !== null);
}

function kernelRows(kernels, showDispatchCounts) {
  if (!Array.isArray(kernels) || !kernels.length) {
    return `<tr><td colspan="${showDispatchCounts ? 4 : 3}" class="profile-missing">No per-kernel timing rows were recorded for this case.</td></tr>`;
  }
  return kernels.map((kernel) => {
    const share = number(kernel.share);
    const shareLabel = share === null ? "Not reported" : `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(Math.abs(share) <= 1 ? share * 100 : share)}%`;
    const dispatched = number(kernel.dispatchedSamples);
    const total = number(kernel.totalSamples);
    const dispatch = dispatched === null && total === null
      ? "Not reported"
      : `Dispatched ${dispatched === null ? "?" : new Intl.NumberFormat().format(dispatched)}/${total === null ? "?" : new Intl.NumberFormat().format(total)} samples`;
    return `<tr><th scope="row">${esc(text(kernel.name, "Unnamed kernel"))}</th><td class="num">${esc(kernelTiming(kernel.medianMs))}</td><td class="num">${esc(shareLabel)}</td>${showDispatchCounts ? `<td>${esc(dispatch)}</td>` : ""}</tr>`;
  }).join("");
}

function caseDetails(testCase) {
  const wall = testCase.wall;
  const gpu = testCase.gpu;
  const showDispatchCounts = hasDispatchCounts(gpu?.kernels);
  const attribution = gpu?.kernelAttribution
    ? `<p class="profile-attribution-note">Per-kernel medians: ${esc(text(gpu.kernelAttribution))}. GPU total is shown separately above.</p>`
    : "";
  return `<details class="profile-case">
    <summary>${esc(caseLabel(testCase))}</summary>
    <div class="profile-case-grid">
      <div><span>Wall latency · unprofiled</span><b>${esc(milliseconds(wall?.medianMs))}</b><small>p95 ${esc(milliseconds(wall?.p95Ms))} · ${esc(sampleCell(wall))}</small></div>
      <div><span>GPU total kernel time · profiled</span><b>${esc(milliseconds(gpu?.medianMs))}</b><small>${esc(sampleCell(gpu))} · diagnostic timing includes instrumentation overhead</small></div>
    </div>
    ${attribution}
    <div class="profile-kernel-table" role="region" aria-label="Per-kernel timings for ${esc(text(testCase.name, "case"))}" tabindex="0">
      <table class="data"><caption>Per-kernel GPU medians · sub-millisecond values shown in µs</caption><thead><tr><th scope="col">Kernel</th><th scope="col" class="num">Median</th><th scope="col" class="num">Share</th>${showDispatchCounts ? "<th scope=\"col\">Dispatched samples</th>" : ""}</tr></thead><tbody>${kernelRows(gpu?.kernels, showDispatchCounts)}</tbody></table>
    </div>
    ${testCase.reason ? `<p class="profile-reason">${esc(testCase.reason)}</p>` : ""}
  </details>`;
}

function packSummary(pack) {
  if (!pack || typeof pack !== "object") return "Not reported";
  const bytes = number(pack.bytes);
  const byteLabel = bytes === null ? "size not reported" : `${new Intl.NumberFormat().format(bytes)} bytes`;
  const hash = text(pack.sha256, "");
  return `${byteLabel}${hash ? ` · SHA-256 ${hash.slice(0, 12)}${hash.length > 12 ? "…" : ""}` : ""}`;
}

function tuningCount(value, unit = "points") {
  const count = number(value);
  if (count !== null) return `${new Intl.NumberFormat().format(count)} ${unit}`;
  if (Array.isArray(value)) return `${value.length} ${value.length === 1 ? unit.replace(/s$/, "") : unit}`;
  return text(value);
}

function selectedCandidateCounts(selected) {
  if (!selected || typeof selected !== "object") return `<span class="profile-missing">Not reported</span>`;
  const entries = Object.entries(selected);
  if (!entries.length) return `<span class="profile-missing">Not reported</span>`;
  return `<span class="profile-count-chips">${entries.slice(0, 4).map(([kind, value]) => `<span><b>${esc(kind)}</b> ${esc(tuningCount(value, "selected"))}</span>`).join("")}${entries.length > 4 ? `<span>+${entries.length - 4} more</span>` : ""}</span>`;
}

function selectionReasonList(reasons) {
  if (!reasons || typeof reasons !== "object") return "";
  const entries = Object.entries(reasons);
  if (!entries.length) return "";
  const shown = entries.slice(0, 5).map(([reason, count]) => `<li><span>${esc(compact(reason, 100))}</span><b>${esc(tuningCount(count, "selected"))}</b></li>`);
  if (entries.length > shown.length) shown.push(`<li class="profile-missing">${entries.length - shown.length} more reasons</li>`);
  return `<ul class="profile-selection-reasons">${shown.join("")}</ul>`;
}

function calibratorSummary(calibrator) {
  const path = text(calibrator.path, "Unnamed calibrator");
  const selection = selectedCandidateCounts(calibrator.selectedCandidates);
  const truncated = calibrator.truncated === true
    ? badge("partial", "Truncated capture")
    : "";
  return `<details class="profile-calibrator">
    <summary><span>${esc(compact(path, 130))}</span><span class="profile-calibrator-summary">${selection}${truncated}</span></summary>
    <div class="profile-calibrator-body">
      ${calibrator.reason ? `<p class="profile-reason"><b>Reason:</b> ${esc(compact(calibrator.reason, 220))}</p>` : ""}
      <dl class="profile-tuning-facts">
        <div><dt>Method</dt><dd>${esc(text(calibrator.method))}</dd></div>
        <div><dt>Elapsed</dt><dd>${esc(milliseconds(calibrator.elapsedMs))}</dd></div>
        <div><dt>Revision</dt><dd>${esc(compact(text(calibrator.revision), 80))}</dd></div>
        <div><dt>Dtype</dt><dd>${esc(text(calibrator.dtype))}</dd></div>
        <div><dt>Shape limit</dt><dd>${esc(text(calibrator.shapeLimit))}</dd></div>
        <div><dt>Available shapes</dt><dd>${esc(tuningCount(calibrator.availableShapes, "shapes"))}</dd></div>
        <div><dt>Scheduled shapes</dt><dd>${esc(tuningCount(calibrator.scheduledShapes, "shapes"))}</dd></div>
        <div><dt>Recorded points</dt><dd>${esc(tuningCount(calibrator.recordedPoints))}</dd></div>
        <div><dt>Projection points</dt><dd>${esc(tuningCount(calibrator.projectionPoints))}</dd></div>
        <div><dt>Measured projection points</dt><dd>${esc(tuningCount(calibrator.measuredProjectionPoints))}</dd></div>
      </dl>
      ${calibrator.selectionReasons ? `<div class="profile-selection-reasons-wrap"><b>Candidate selection reasons</b>${selectionReasonList(calibrator.selectionReasons)}</div>` : ""}
    </div>
  </details>`;
}

function tuningSummaryDetails(value) {
  const groups = (Array.isArray(value) ? value : value && typeof value === "object" ? [value] : [])
    .filter((group) => group && typeof group === "object");
  if (!groups.length) return "";
  const count = groups.reduce((total, group) => total + (Array.isArray(group.calibrators) ? group.calibrators.length : 0), 0);
  return `<details class="profile-tuning-summary">
    <summary>Kernel tuning summary · ${count} calibrator${count === 1 ? "" : "s"}</summary>
    <div class="profile-tuning-groups">
      ${groups.map((group) => {
        const calibrators = Array.isArray(group.calibrators) ? group.calibrators : [];
        const truncated = group.summaryTruncated === true ? `<span>${badge("partial", "Summary truncated")}</span>` : "";
        return `<section class="profile-tuning-group">
          <header><b>${esc(text(group.block, "Block not reported"))}</b><span>Profile: ${esc(text(group.profile))}</span>${truncated}</header>
          ${calibrators.length ? `<div class="profile-calibrator-list">${calibrators.slice(0, 8).map(calibratorSummary).join("")}</div>${calibrators.length > 8 ? `<p class="profile-missing">Showing 8 of ${calibrators.length} calibrators. See raw summary for the full list.</p>` : ""}` : `<p class="profile-missing">No calibrator details were recorded.</p>`}
        </section>`;
      }).join("")}
    </div>
  </details>`;
}

function modelDetails(model) {
  const cases = Array.isArray(model.cases) ? model.cases : [];
  const reason = model.reason ? `<p class="profile-reason"><b>${esc(text(model.status))}:</b> ${esc(model.reason)}</p>` : "";
  return `<div class="profile-model-detail">
    <p class="profile-pack"><b>Pack:</b> ${esc(packSummary(model.pack))}</p>
    ${reason}
    ${tuningSummaryDetails(model.tuningSummary)}
    ${cases.length ? `<div class="profile-case-list">${cases.map(caseDetails).join("")}</div>` : `<p class="profile-missing">No case measurements were recorded. ${esc(text(model.reason, "Reason not reported."))}</p>`}
  </div>`;
}

function modelRow(model, environment) {
  const name = text(model.name, text(model.id, "Unnamed model"));
  const id = model.id && model.name ? `<small>${esc(model.id)}</small>` : "";
  const cases = Array.isArray(model.cases) ? model.cases : [];
  const status = text(model.status, "Not reported");
  const browser = browserLabel(model.browser ?? environment.browser);
  const precision = text(model.precision, "Not reported");
  const reason = model.reason ? `<small class="profile-model-reason">${esc(model.reason)}</small>` : "";
  return `<tr>
    <th scope="row"><details class="profile-model"><summary><span>${esc(name)}</span>${id}</summary>${modelDetails(model)}</details></th>
    <td>${esc(precision)}</td><td>${esc(browser)}</td>
    <td>${badge(status)}${reason}</td>
    <td class="profile-timing-cell">${metricList(model, "wall")}</td>
    <td class="profile-timing-cell">${metricList(model, "gpu")}</td>
  </tr>`;
}

function gateCounts(rows, key) {
  const counts = new Map();
  for (const row of rows) {
    const value = text(row[key], "Not reported");
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
}

function gateSummary(rows, key, label) {
  const counts = gateCounts(rows, key);
  const content = counts.length
    ? counts.map(([state, count]) => `${badge(state, `${state}: ${count}`)}`).join(" ")
    : badge("Not reported");
  return `<div class="profile-gate"><b>${esc(label)}</b><div>${content}</div></div>`;
}

function kernelTable(rows) {
  if (!rows.length) return `<p class="profile-missing">No kernel result rows were included in this report.</p>`;
  return `<div class="profile-kernel-table" role="region" aria-label="Kernel validation and timing status" tabindex="0">
    <table class="data"><thead><tr><th scope="col">Kernel</th><th scope="col">Overall</th><th scope="col">Compile</th><th scope="col">Numerical</th><th scope="col">Timing</th><th scope="col">Harnesses</th><th scope="col">Notes</th><th scope="col">Browser-specific gates</th></tr></thead>
    <tbody>${rows.map((row) => `<tr>
      <th scope="row">${esc(text(row.name, "Unnamed kernel"))}</th>
      <td>${badge(row.status)}</td><td>${badge(row.compileStatus)}</td><td>${badge(row.numericalStatus)}</td><td>${badge(row.timingStatus)}</td>
      <td>${esc(text(row.harnesses))}</td><td>${esc(text(row.reason))}</td><td>${browserGateDetails(row.browsers)}</td>
    </tr>`).join("")}</tbody></table>
  </div>`;
}

function evidenceItems(items) {
  if (!Array.isArray(items) || !items.length) return `<span class="profile-missing">No evidence entries reported.</span>`;
  const shown = items.slice(0, 4).map((item) => {
    const value = typeof item === "string" ? item : compact(item, 180);
    const url = typeof item === "object" && item ? safeUrl(item.url) : safeUrl(value);
    const label = typeof item === "object" && item ? text(item.label ?? item.name ?? item.url, "Evidence") : value;
    return `<li>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} ↗</a>` : esc(value)}</li>`;
  });
  if (items.length > shown.length) shown.push(`<li class="profile-missing">${items.length - shown.length} more entries are in the raw summary.</li>`);
  return `<ul class="profile-evidence-list">${shown.join("")}</ul>`;
}

function browserGateDetails(browsers) {
  if (!Array.isArray(browsers) || !browsers.length) return `<span class="profile-missing">Not reported</span>`;
  const limit = 6;
  const shown = browsers.slice(0, limit);
  return `<details class="profile-browser-gates">
    <summary>${shown.length} browser ${shown.length === 1 ? "result" : "results"}</summary>
    <p class="profile-browser-note">Per-browser outcomes are shown separately because adapter features can differ between browsers.</p>
    <div class="profile-browser-list">
      ${shown.map((browser) => `<article class="profile-browser-result">
        <div class="profile-browser-head"><b>${esc(browserLabel(browser.browser ?? browser.name))}</b>${badge(browser.status)}</div>
        <div class="profile-browser-statuses"><span>Compile ${badge(browser.compileStatus)}</span><span>Numerical ${badge(browser.numericalStatus)}</span><span>Timing ${badge(browser.timingStatus)}</span></div>
        <dl class="profile-browser-facts"><div><dt>Compilation</dt><dd>${esc(text(browser.compilation))}</dd></div><div><dt>Isolated</dt><dd>${esc(text(browser.isolated))}</dd></div><div><dt>Harnesses</dt><dd>${esc(text(browser.harnesses))}</dd></div></dl>
        ${browser.reason ? `<p class="profile-reason">${esc(browser.reason)}</p>` : ""}
        <div class="profile-browser-evidence"><b>Evidence</b>${evidenceItems(browser.evidence)}</div>
      </article>`).join("")}
    </div>
    ${browsers.length > limit ? `<p class="profile-browser-note">Showing ${limit} of ${browsers.length} browser results. See the raw summary for the full list.</p>` : ""}
  </details>`;
}

function improvementCard(improvement) {
  const before = improvement.beforeLabel || kernelTiming(improvement.beforeMs);
  const after = improvement.afterLabel || kernelTiming(improvement.afterMs);
  const change = number(improvement.changePercent);
  const formattedChange = change === null ? "Change not reported" : `${change > 0 ? "+" : ""}${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(change)}%`;
  const changeLabel = improvement.changeLabel || formattedChange;
  return `<article class="profile-improvement card">
    <div class="profile-improvement-head"><h4>${esc(text(improvement.name, "Tuning change"))}</h4>${badge(improvement.status)}</div>
    <p class="muted small">${esc(text(improvement.scope, "Scope not reported"))}</p>
    <div class="profile-improvement-values"><span>Before <b>${esc(before)}</b></span><span>After <b>${esc(after)}</b></span><strong>${esc(changeLabel)}</strong></div>
    ${improvement.reason ? `<p class="profile-reason">${esc(improvement.reason)}</p>` : ""}
  </article>`;
}

function comparisonChange(value) {
  const change = number(value);
  if (change === null) return { label: "Change not reported", tone: "neutral" };
  if (Math.abs(change) < 0.05 || change === 0) return { label: "0%", tone: "neutral" };
  return {
    label: `${change > 0 ? "+" : ""}${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(change)}%`,
    tone: change > 0 ? "worse" : "better",
  };
}

function comparisonMetric(metric) {
  if (!metric || typeof metric !== "object") return `<span class="profile-missing">Not measured</span>`;
  const change = comparisonChange(metric.changePercent);
  return `<div class="profile-improvement-values profile-optimization-values">
    <span>Before <b>${esc(kernelTiming(metric.beforeMs))}</b></span>
    <span>After <b>${esc(kernelTiming(metric.afterMs))}</b></span>
    <strong class="profile-comparison-change ${change.tone}">${esc(change.label)}</strong>
  </div>`;
}

function probabilityDelta(value) {
  const delta = number(value);
  return delta === null ? "Not reported" : new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(delta);
}

function optimizationModelRows(models, environment) {
  const rows = [];
  for (const model of models) {
    const cases = Array.isArray(model.cases) ? model.cases : [];
    const name = text(model.name, text(model.id, "Unnamed model"));
    const id = model.id && model.name ? `<small>${esc(model.id)}</small>` : "";
    const precision = text(model.precision, "Precision not reported");
    const browser = browserLabel(model.browser ?? environment.browser);
    const delta = probabilityDelta(model.correctness?.maxProbabilityDelta);
    const probabilitySource = text(model.correctness?.probabilitySource, "Source not reported");
    const probabilityCell = `<td class="profile-probability-delta"><span class="num">${esc(delta)}</span><small>${esc(probabilitySource)}</small></td>`;
    const metrics = (testCase) => `<td>${comparisonMetric(testCase?.wall)}</td><td>${comparisonMetric(testCase?.gpu)}</td>`;
    if (!cases.length) {
      rows.push(`<tr><th scope="row"><span>${esc(name)}</span>${id}<small>${esc(precision)}</small></th><td>${esc(browser)}</td><td class="profile-missing">No case comparisons recorded.</td>${probabilityCell}<td><span class="profile-missing">Not measured</span></td><td><span class="profile-missing">Not measured</span></td></tr>`);
      continue;
    }
    for (const testCase of cases) {
      rows.push(`<tr><th scope="row"><span>${esc(name)}</span>${id}<small>${esc(precision)}</small></th><td>${esc(browser)}</td><td>${esc(text(testCase.name, "Case not reported"))}</td>${probabilityCell}${metrics(testCase)}</tr>`);
    }
  }
  return rows.length ? rows.join("") : `<tr><td colspan="6" class="profile-missing">No model comparisons were recorded.</td></tr>`;
}

function sourceSnapshot(label, source, commitLabel) {
  return `<div><dt>${esc(label)}</dt><dd>${esc(commitLabel)} <code>${esc(text(source?.commit ?? source?.baseCommit))}</code></dd><dd>Source fingerprint <code>${esc(text(source?.fingerprint))}</code></dd></div>`;
}

function artifactDetails(artifacts) {
  if (!Array.isArray(artifacts) || !artifacts.length) return `<p class="profile-missing">No comparison artifacts were reported.</p>`;
  return `<details class="profile-optimization-artifacts">
    <summary>Measurement artifacts (${artifacts.length})</summary>
    <ul>${artifacts.map((artifact) => {
      const url = safeUrl(artifact.url);
      const label = text(artifact.label, "Report artifact");
      const size = number(artifact.bytes);
      const gzipSize = number(artifact.gzipBytes);
      const hashes = [artifact.sha256 ? `SHA-256 ${text(artifact.sha256)}` : "", artifact.gzipSha256 ? `gzip SHA-256 ${text(artifact.gzipSha256)}` : ""].filter(Boolean).join(" · ");
      return `<li>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} ↗</a>` : esc(label)}<small>${size === null ? "Size not reported" : `${new Intl.NumberFormat().format(size)} bytes`}${gzipSize === null ? "" : ` · gzip ${new Intl.NumberFormat().format(gzipSize)} bytes`}${hashes ? ` · ${esc(hashes)}` : ""}</small></li>`;
    }).join("")}</ul>
  </details>`;
}

function renderModelOptimization(report) {
  const environment = report.environment ?? {};
  const improvements = Array.isArray(report.improvements) ? report.improvements : [];
  const models = Array.isArray(report.models) ? report.models : [];
  const caseCount = models.reduce((count, model) => count + (Array.isArray(model.cases) ? model.cases.length : 0), 0);
  const modelCount = new Set(models.map((model) => text(model.id ?? model.name, "")).filter(Boolean)).size;
  const configurationCount = models.length;
  const before = report.sources?.before ?? {};
  const after = report.sources?.after ?? {};
  return `<section class="profile-optimization">
    <header class="profile-heading">
      <div><div class="eyebrow">Measured optimization results</div><h3>Before and after model latency</h3>
        <p class="muted">End-to-end results apply to this model pack, hardware, browser, and source capture. Lower latency is better.</p>
      </div>
      <div class="profile-links">
        <span class="profile-optimization-status">${badge(report.status, `Report ${text(report.status)}`)}</span>
        <div class="profile-primary-links"><a href="${RAW_OPTIMIZATION_REPORT_URL}" target="_blank" rel="noopener">Raw summary JSON ↗</a><a href="${METHOD_URL}" target="_blank" rel="noopener">Measurement method ↗</a></div>
      </div>
    </header>
    <div class="profile-optimization-facts">
      <div><span>Hardware</span><b>${esc(text(environment.hardware))}</b></div>
      <div><span>Browser</span><b>${esc(browserLabel(environment.browser))}</b></div>
      <div><span>Capture</span><b>${esc(dateLabel(report.generatedAt))}</b></div>
      <div><span>Model cases</span><b>${caseCount}</b></div>
    </div>
    <p class="profile-method-note"><b>Read the measures separately:</b> wall latency is the end-to-end measure. GPU time is diagnostic. A negative latency change means the measured time decreased. ${esc(compact(text(report.protocol?.description, "Protocol details are in the report summary."), 300))}</p>
    <p class="profile-separation-note">The model and kernel profile below uses separate fixtures and a separate source fingerprint. Its latency values are not paired before-and-after comparisons for the optimization results above.</p>
    <section class="profile-subsection">
      <div class="profile-section-head"><div><h4>Optimization changes tested</h4><p class="muted small">Each card reports its scope, outcome status, and measured values when available.</p></div></div>
      ${improvements.length ? `<div class="grid-2 profile-improvements">${improvements.map(improvementCard).join("")}</div>` : `<p class="profile-missing">No individual optimization changes were reported.</p>`}
    </section>
    <section class="profile-subsection profile-optimization-comparisons">
      <details class="profile-disclosure"><summary>Show comparison table · ${modelCount} ${modelCount === 1 ? "model" : "models"} · ${caseCount} ${caseCount === 1 ? "case" : "cases"}${configurationCount !== modelCount ? ` · ${configurationCount} configurations` : ""}</summary>
        <div class="profile-optimization-table-wrap" role="region" aria-label="Before and after model latency comparisons" tabindex="0">
          <table class="data profile-optimization-table"><caption>Recorded maximum |Δp| · source labels distinguish raw distributions from rounded public values</caption>
            <thead><tr><th scope="col">Model · precision</th><th scope="col">Browser</th><th scope="col">Case</th><th scope="col">Recorded |Δp|</th><th scope="col">Wall latency</th><th scope="col">GPU time</th></tr></thead>
            <tbody>${optimizationModelRows(models, environment)}</tbody>
          </table>
        </div>
      </details>
    </section>
    <section class="profile-subsection profile-optimization-provenance">
      <h4>Capture and source provenance</h4>
      <p class="muted small">Fingerprints identify the measured source files. The after capture's commit field is its base commit, not a claim that the checkout was clean.</p>
      <dl class="profile-optimization-sources">${sourceSnapshot("Before capture", before, "Commit")}${sourceSnapshot("After capture", after, "Base commit")}</dl>
      <details class="profile-disclosure"><summary>Benchmark protocol and artifacts</summary>
        <div class="profile-optimization-details"><div><h4>Protocol</h4>${detailList(report.protocol)}</div><div><h4>Artifacts</h4>${artifactDetails(report.artifacts)}</div></div>
      </details>
      ${limitations(report.limitations)}
    </section>
  </section>`;
}

function detailList(data) {
  if (!data || typeof data !== "object") return `<p class="profile-missing">Not reported.</p>`;
  const entries = Object.entries(data);
  if (!entries.length) return `<p class="profile-missing">No details were recorded.</p>`;
  return `<dl class="profile-kv">${entries.map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(text(value))}</dd></div>`).join("")}</dl>`;
}

function artifactLinks(artifacts) {
  const entries = Array.isArray(artifacts) ? artifacts : [];
  const excluded = new Set([safeUrl(RAW_REPORT_URL), safeUrl(METHOD_URL)]);
  const seen = new Set(excluded);
  const links = entries.map((item) => {
    const href = safeUrl(item?.url);
    if (!href || seen.has(href)) return "";
    seen.add(href);
    return `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(text(item.label, "Report artifact"))} ↗</a>`;
  }).filter(Boolean);
  const limit = 12;
  const shown = links.slice(0, limit);
  const extra = links.length > limit ? `<p class="profile-missing">Showing ${limit} of ${links.length}; the raw summary has the complete artifact list.</p>` : "";
  return `<div class="profile-primary-links"><a href="${RAW_REPORT_URL}" target="_blank" rel="noopener">Raw summary JSON ↗</a><a href="${METHOD_URL}" target="_blank" rel="noopener">Measurement method ↗</a></div>
    ${links.length ? `<details class="profile-artifact-disclosure"><summary>Raw evidence (${links.length})</summary><ul>${shown.map((link) => `<li>${link}</li>`).join("")}</ul>${extra}</details>` : ""}`;
}

function coverage(report) {
  const models = report.models;
  const modelId = (model) => text(model.id ?? model.name, "");
  const catalogIds = new Set(models.map(modelId).filter(Boolean));
  const measuredRows = models.filter((model) => model.status === "measured");
  const measuredModelIds = new Set(measuredRows.map(modelId).filter(Boolean));
  const measuredCases = measuredRows.flatMap((model) => Array.isArray(model.cases) ? model.cases : []);
  const timedCases = measuredCases.filter((testCase) => measuredMetric(testCase.wall) || measuredMetric(testCase.gpu)).length;
  const rows = report.kernels.rows;
  const kernelRowsMeasured = rows.filter((row) => row.status === "measured" || (Array.isArray(row.browsers) && row.browsers.some((browser) => browser?.status === "measured"))).length;
  const registered = number(report.kernels.registered);
  const countText = registered === null ? `${rows.length} rows` : `${rows.length} / ${registered}`;
  return `<div class="grid-4 tiles profile-coverage">
    <div class="tile"><span class="tl">Profile status</span><b>${badge(report.status)}</b><span class="ts">Generated ${esc(dateLabel(report.generatedAt))}</span></div>
    <div class="tile"><span class="tl">Models measured</span><b>${measuredModelIds.size} <small>/ ${catalogIds.size}</small></b><span class="ts">${measuredRows.length} / ${models.length} model configuration rows measured</span></div>
    <div class="tile"><span class="tl">Timed case results</span><b>${timedCases}</b><span class="ts">Across measured configurations only; unavailable configurations are excluded</span></div>
    <div class="tile"><span class="tl">Kernel rows reported</span><b>${esc(countText)}</b><span class="ts">${kernelRowsMeasured} checked and timed on at least one browser; per-browser gaps are listed below</span></div>
  </div>`;
}

function metadata(report) {
  const env = report.environment ?? {};
  const runtime = report.runtime ?? {};
  const commit = text(runtime.commit, "Not reported");
  const fingerprint = text(runtime.sourceFingerprint, "Not reported");
  const hostGpus = env.hostGpus;
  const hasHostGpus = Array.isArray(hostGpus)
    ? hostGpus.length > 0
    : hostGpus && typeof hostGpus === "object"
      ? Object.keys(hostGpus).length > 0
      : hostGpus !== undefined && hostGpus !== null && String(hostGpus).trim() !== "";
  const gpuLabel = hasHostGpus ? "Reported host GPU hardware" : "GPU adapter";
  const gpuName = hasHostGpus ? text(hostGpus) : text(env.gpu);
  const adapterName = hasHostGpus && env.gpu ? `<small>Browser adapter: ${esc(text(env.gpu))}</small>` : "";
  const shortCommit = commit.length > 12 ? `${commit.slice(0, 12)}…` : commit;
  return `<div class="profile-meta-grid">
    <div><span>${gpuLabel}</span><b>${esc(gpuName)}</b>${adapterName}</div>
    <div><span>Browser</span><b>${esc(browserLabel(env.browser))}</b></div>
    <div><span>Operating system</span><b>${esc(text(env.os))}</b></div>
    <div><span>Measured runtime snapshot</span><b title="Measured source fingerprint"><code>${esc(fingerprint)}</code></b><small title="${esc(commit)}">Base commit ${esc(shortCommit)}</small></div>
  </div>`;
}

function sourceCaptureNote(runtime) {
  if (runtime.workingTreeDirty !== true) return "";
  const commit = text(runtime.commit, "Not reported");
  return `<p class="profile-dirty-source"><b>Captured before landing:</b> this run included uncommitted runtime changes based on commit <code>${esc(commit)}</code>. Raw source fingerprints in the summary identify the files included in this capture.</p>`;
}

function limitations(items) {
  if (!Array.isArray(items) || !items.length) return "";
  return `<div class="profile-limitations"><h4>Limits of this run</h4><ul>${items.map((item) => `<li>${esc(text(item))}</li>`).join("")}</ul></div>`;
}

function renderReport(report) {
  const models = report.models;
  const kernels = report.kernels.rows;
  const improvements = Array.isArray(report.improvements) ? report.improvements : [];
  const env = report.environment ?? {};
  return `<div class="kernel-profile">
    <header class="profile-heading">
      <div><div class="eyebrow">Latest measured profile</div><h3>Current model latency and kernel coverage</h3>
        <p class="muted">All 11 Q8 models completed. One kernel harness failed during browser device creation; a separate recheck passed. The original failure remains recorded.</p>
      </div>
      <div class="profile-links">${artifactLinks(report.artifacts)}</div>
    </header>
    ${coverage(report)}
    ${metadata(report)}
    ${sourceCaptureNote(report.runtime ?? {})}
    <p class="profile-method-note"><b>Read timings separately:</b> wall latency comes from unprofiled runs. GPU kernel timings come from instrumented runs and include profiling overhead. Do not subtract GPU time from wall time or compare the two as if they were measured in the same run.</p>
    <div class="profile-table-wrap reference-table" role="region" aria-label="Current model latency by case" tabindex="0">
      <table class="data profile-model-table"><caption>Current model results · wall latency and profiled GPU time are separate measurements</caption>
        <thead><tr><th scope="col">Model</th><th scope="col">Precision</th><th scope="col">Browser</th><th scope="col">Status</th><th scope="col">Wall latency</th><th scope="col">Profiled GPU time</th></tr></thead>
        <tbody>${models.map((model) => modelRow(model, env)).join("")}</tbody>
      </table>
    </div>
    <section class="profile-subsection">
      <div class="profile-section-head"><div><h4>Kernel validation and timing</h4><p class="muted small">Each row reports the compile, numerical, and timing status supplied by the run.</p></div></div>
      <div class="profile-gates">${gateSummary(kernels, "compileStatus", "Compile")}${gateSummary(kernels, "numericalStatus", "Numerical")}${gateSummary(kernels, "timingStatus", "Timing")}</div>
      <details class="profile-disclosure"><summary>Show ${kernels.length} kernel rows</summary>${kernelTable(kernels)}</details>
    </section>
    <section class="profile-subsection">
      <div class="profile-section-head"><div><h4>Changes tested on this adapter</h4><p class="muted small">Reported before and after values are tied to this hardware, driver, browser, and build.</p></div></div>
      ${improvements.length ? `<div class="grid-2 profile-improvements">${improvements.map(improvementCard).join("")}</div>` : `<p class="profile-missing">No tuning comparisons were recorded in this report.</p>`}
    </section>
    <section class="profile-subsection profile-run-details">
      <details class="profile-disclosure"><summary>Environment and benchmark protocol</summary>
        <div class="profile-detail-grid"><div><h4>Environment</h4>${detailList(env)}</div><div><h4>Protocol</h4>${detailList(report.protocol)}</div></div>
      </details>
      ${limitations(report.limitations)}
    </section>
  </div>`;
}

function validateReport(report) {
  if (!report || typeof report !== "object" || report.schemaVersion !== 1) return "Unsupported or missing schemaVersion; expected version 1.";
  if (!Array.isArray(report.models)) return "The summary does not contain a models array.";
  if (!report.kernels || !Array.isArray(report.kernels.rows)) return "The summary does not contain kernel result rows.";
  return "";
}

function unavailable(reason) {
  return `<section class="profile-unavailable" role="status">
    <div class="eyebrow">Latest measured profile</div><h3>Report unavailable</h3>
    <p>The current model and kernel measurements are not available from the published summary file.</p>
    <p class="profile-unavailable-reason"><b>Detail:</b> ${esc(reason)}</p>
    <div class="profile-links"><a href="${RAW_REPORT_URL}" target="_blank" rel="noopener">Check the raw summary file ↗</a><a href="${METHOD_URL}" target="_blank" rel="noopener">Measurement method ↗</a></div>
  </section>`;
}

function optimizationUnavailable(reason) {
  return `<section class="profile-unavailable profile-optimization-unavailable" role="status">
    <div class="eyebrow">Measured optimization results</div><h3>Comparison unavailable</h3>
    <p>The before-and-after model comparison is not available from the published summary file.</p>
    <p class="profile-unavailable-reason"><b>Detail:</b> ${esc(reason)}</p>
    <div class="profile-links"><a href="${RAW_OPTIMIZATION_REPORT_URL}" target="_blank" rel="noopener">Check the raw comparison summary ↗</a><a href="${METHOD_URL}" target="_blank" rel="noopener">Measurement method ↗</a></div>
  </section>`;
}

function validateOptimizationReport(report) {
  if (!report || typeof report !== "object" || report.schemaVersion !== 1) return "Unsupported or missing schemaVersion; expected version 1.";
  if (!Array.isArray(report.models)) return "The summary does not contain a models array.";
  return "";
}

export function mountModelOptimization(container) {
  if (!container) return () => {};
  const controller = new AbortController();
  container.innerHTML = `<p class="profile-loading" role="status">Loading measured optimization results…</p>`;
  fetch(new URL(OPTIMIZATION_REPORT_PATH, document.baseURI), { signal: controller.signal, cache: "no-cache" })
    .then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status} while loading ${OPTIMIZATION_REPORT_PATH}`);
      return response.json();
    })
    .then((report) => {
      const problem = validateOptimizationReport(report);
      if (problem) throw new Error(problem);
      container.innerHTML = renderModelOptimization(report);
    })
    .catch((error) => {
      if (error?.name === "AbortError") return;
      container.innerHTML = optimizationUnavailable(error?.message || "The comparison could not be loaded or parsed.");
    });
  return () => controller.abort();
}

export function mountKernelProfile(container) {
  if (!container) return () => {};
  const controller = new AbortController();
  container.innerHTML = `<p class="profile-loading" role="status">Loading the latest model and kernel profile…</p>`;
  fetch(new URL(REPORT_PATH, document.baseURI), { signal: controller.signal, cache: "no-cache" })
    .then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status} while loading ${REPORT_PATH}`);
      return response.json();
    })
    .then((report) => {
      const problem = validateReport(report);
      if (problem) throw new Error(problem);
      container.innerHTML = renderReport(report);
    })
    .catch((error) => {
      if (error?.name === "AbortError") return;
      container.innerHTML = unavailable(error?.message || "The report could not be loaded or parsed.");
    });
  return () => controller.abort();
}
