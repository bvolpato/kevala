#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "selenium>=4.25,<5",
#   "websockets>=13",
# ]
# [tool.uv]
# required-version = ">=0.13.4"
# exclude-newer = "P7D"
# ///
"""Run a deterministic browser benchmark in an isolated headless browser, or in a running Chrome (--cdp).

The page owns timing and correctness checks. This wrapper only controls the
browser, validates the requested backend, saves the structured result, and
prints the scalar metric last so it can be consumed by an autoresearch loop.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


DECISION_PROBABILITY_SUM_TOLERANCE = 0.0005
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CHROME_DIAGNOSTIC_BUDGET_BYTES = 32 * 1024**2
Q8_TERMINAL_EXPRESSION = "return Boolean(window.bench && window.bench.done === true);"
Q8_RESULT_EXPRESSION = "return window.bench || null;"


def poll_q8_page(execute_script, deadline: float, check_diagnostics, *, clock=time.monotonic, wait=time.sleep):
    """Poll a tiny terminal flag and fetch the full page result once."""
    while clock() < deadline:
        check_diagnostics("poll")
        if execute_script(Q8_TERMINAL_EXPRESSION) is True:
            check_diagnostics("terminal-before-result-fetch")
            result = execute_script(Q8_RESULT_EXPRESSION)
            check_diagnostics("after-result-fetch")
            return True, result
        wait(0.25)
    return False, None


def thread_identity_delta(previous: list[dict], current: list[dict], expected_cgroup: str) -> dict:
    """Compare sequential TID samples and report verified thread churn."""
    def index(rows: list[dict], label: str) -> tuple[dict[tuple[int, int], dict], dict[int, str]]:
        by_tid: dict[tuple[int, int], dict] = {}
        process_starts: dict[int, str] = {}
        for row in rows:
            pid, tid = row.get("pid"), row.get("tid")
            process_start, process_after = row.get("processStartTicks"), row.get("processStartTicksAfter")
            if not isinstance(pid, int) or not isinstance(tid, int) or not isinstance(row.get("startTicks"), str):
                raise ValueError(f"Q8 SwiGLU {label} thread sample has an invalid PID/TID identity")
            if row.get("cgroup") != expected_cgroup or row.get("cgroupAfter") != expected_cgroup:
                raise ValueError(f"Q8 SwiGLU {label} TID {tid} escaped its exact cgroup")
            if not isinstance(process_start, str) or process_after != process_start:
                raise ValueError(f"Q8 SwiGLU {label} PID {pid} changed identity during its TID inventory")
            prior_process_start = process_starts.setdefault(pid, process_start)
            if prior_process_start != process_start:
                raise ValueError(f"Q8 SwiGLU {label} PID {pid} identity was reused within its TID inventory")
            key = (pid, tid)
            if key in by_tid:
                raise ValueError(f"Q8 SwiGLU {label} duplicated TID {tid} for PID {pid}")
            by_tid[key] = row
        return by_tid, process_starts

    before, before_process_starts = index(previous, "previous")
    after, after_process_starts = index(current, "current")
    for pid in before_process_starts.keys() & after_process_starts.keys():
        if before_process_starts[pid] != after_process_starts[pid]:
            raise ValueError(f"Q8 SwiGLU PID {pid} identity was reused between TID inventories")
    for key in before.keys() & after.keys():
        prior, observed = before[key], after[key]
        if prior["processStartTicks"] != observed["processStartTicks"]:
            raise ValueError(f"Q8 SwiGLU PID/TID identity was reused for PID {key[0]} TID {key[1]}")
        if prior["startTicks"] != observed["startTicks"]:
            raise ValueError(f"Q8 SwiGLU TID {key[1]} identity was reused for PID {key[0]}")
    new_keys = after.keys() - before.keys()
    disappeared_keys = before.keys() - after.keys()
    return {
        "newThreads": [after[key] for key in sorted(new_keys)],
        "disappearedThreads": [before[key] for key in sorted(disappeared_keys)],
        "stableThreadCount": len(before.keys() & after.keys()),
    }


def _q8_check_original_anchors(original_anchors: dict, read_process, phase) -> dict:
    anchors = {}
    for anchor_name, original in original_anchors.items():
        phase("check-original-anchor", anchorName=anchor_name, pid=original["pid"])
        try:
            current = read_process(original["pid"])
            same = current["startTicks"] == original["startTicks"]
            anchors[anchor_name] = {"state": "same-process-alive" if same else "identity-reused", "current": current}
            if same:
                raise ValueError(f"Q8 SwiGLU {anchor_name} process survived browser quit")
        except (FileNotFoundError, ProcessLookupError):
            anchors[anchor_name] = {"state": "absent"}
    return anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        choices=("gpu", "bench", "latency", "parity", "decision", "kernels", "cache", "tetris", "q8-swiglu"),
        default="gpu",
        help="page result contract to validate",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="benchmark page URL, including query parameters",
    )
    parser.add_argument(
        "--backend",
        choices=("webgpu", "wasm"),
        default="webgpu",
        help="backend expected from the page (default: webgpu)",
    )
    parser.add_argument(
        "--browser",
        choices=("firefox", "chromium"),
        default="firefox",
        help="browser to run (default: firefox)",
    )
    parser.add_argument(
        "--binary",
        "--browser-binary",
        dest="binary",
        help="browser executable; otherwise use FIREFOX_BIN or discover Chromium",
    )
    parser.add_argument("--timeout", type=float, default=240.0, help="maximum browser wait in seconds")
    parser.add_argument("--output", type=Path, help="write the structured page result to this JSON file")
    parser.add_argument("--max-dp", type=float, default=0.024, help="maximum parity probability difference")
    parser.add_argument("--headed", action="store_true", help="show a benchmark window instead of a headless browser or hidden CDP target")
    parser.add_argument("--chromium-arg", action="append", default=[], help="extra argument for the isolated Chromium process; repeat as --chromium-arg=--flag")
    parser.add_argument(
        "--cdp",
        default=os.environ.get("KEVALA_BENCH_CDP"),
        help="run in a hidden target of the Chrome listening at this DevTools URL instead of launching a browser",
    )
    parser.add_argument(
        "--cdp-default-context",
        action="store_true",
        help="use an owned target in Chrome's default context to avoid hidden-target context teardown crashes",
    )
    parser.add_argument("--q8-swiglu-session-bus-helper", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--q8-swiglu-scope-unit", help=argparse.SUPPRESS)
    parser.add_argument("--chrome-diagnostics-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


DEFAULT_URLS = {
    "gpu": "http://127.0.0.1:18086/dev/gpu-bench.html",
    "bench": "http://127.0.0.1:18086/bench.html#auto&backend=webgpu&pack=local&model=laya&profile=1&runs=10&unique=1",
    "latency": "http://127.0.0.1:18086/bench.html#auto&backend=webgpu&pack=local&model=laya&profile=0&runs=10&unique=1",
    "parity": "http://127.0.0.1:18086/parity.html#auto&backend=webgpu&pack=local",
    "kernels": "http://127.0.0.1:18086/dev/kernels.html",
    "cache": "http://127.0.0.1:18086/dev/cache-test.html?backend=webgpu",
    "tetris": "http://127.0.0.1:18086/dev/tetris-eval.html#model=kev-4b&backend=webgpu&pieces=20&seeds=1,2,3",
    "decision": "http://127.0.0.1:18086/dev/decision-bench.html?model=laya&backend=webgpu&pack=local&permutations=3&dataset=all",
    "q8-swiglu": "http://127.0.0.1:18086/bench.html#auto&backend=webgpu&pack=local&model=kev-9b&precision=q8&onPage=1&cache=0&stateCache=0&profile=0&gpuKernel=auto&submit=await&swigluTrial=1",
}


RESULT_GLOBALS = {
    "gpu": "gpuBench",
    "bench": "bench",
    "latency": "bench",
    "parity": "parity",
    "kernels": "kernelResults",
    "cache": "ct",
    "tetris": "evalResult",
    "decision": "decisionBench",
    "q8-swiglu": "bench",
}


def url_with_backend(url: str, backend: str) -> str:
    """Set a benchmark backend in either a query string or a hash query."""
    parts = urlsplit(url)
    if parts.fragment:
        pieces = parts.fragment.split("&")
        for index, piece in enumerate(pieces):
            if piece == "backend" or piece.startswith("backend="):
                pieces[index] = f"backend={backend}"
                return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "&".join(pieces)))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, f"{parts.fragment}&backend={backend}"))
    params = parse_qsl(parts.query, keep_blank_values=True)
    replaced = False
    output = []
    for key, value in params:
        if key == "backend":
            if not replaced:
                output.append((key, backend))
                replaced = True
        else:
            output.append((key, value))
    if not replaced:
        output.append(("backend", backend))
    updated = urlencode(output)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, updated, parts.fragment))


def result_is_done(result: object, kind: str) -> bool:
    if kind == "q8-swiglu":
        return result is True or isinstance(result, dict) and (result.get("done") is True or bool(result.get("error")))
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return True
    if "done" in result:
        return result.get("done") is True
    if "status" in result:
        return result.get("status") in {"done", "error"}
    if kind in {"gpu", "kernels"}:
        # these pages report `status` or `done`; without either the page is still running
        return False
    if kind == "tetris":
        return isinstance(result.get("summary"), dict) and isinstance(result.get("games"), list)
    return isinstance(result.get("backend"), str)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_backend(result: dict, expected: str) -> int | None:
    """Validate the page's selected backend and return its reported thread count."""
    backend = result.get("backend")
    if not isinstance(backend, str):
        raise ValueError(f"result backend is {backend!r}, expected {expected!r}")
    reported = result.get("threads")
    if reported is not None and (not isinstance(reported, int) or isinstance(reported, bool) or reported <= 0):
        raise ValueError(f"result threads must be a positive integer, got {reported!r}")
    if expected == "webgpu":
        if backend != "webgpu":
            raise ValueError(f"result backend is {backend!r}, expected 'webgpu'")
        return reported
    if reported is None:
        raise ValueError("wasm result did not report a positive thread count")
    match = re.fullmatch(r"wasm-([^ ]+)(?: x([0-9]+))?", backend)
    if not match:
        raise ValueError(f"result backend is {backend!r}, expected a wasm-* backend")
    actual_threads = int(match.group(2) or "1")
    if actual_threads <= 0:
        raise ValueError(f"result backend reports invalid thread count: {backend!r}")
    if reported != actual_threads:
        raise ValueError(f"result threads {reported} disagree with backend {backend!r}")
    return reported


def bench_metric(result: dict, expected_backend: str = "webgpu") -> float:
    validate_backend(result, expected_backend)
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark has no cases")
    medians: list[float] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("samples"), list) or not case["samples"]:
            raise ValueError(f"benchmark case {index} has no profiled samples")
        totals: list[float] = []
        for sample_index, sample in enumerate(case["samples"]):
            if not isinstance(sample, dict):
                raise ValueError(f"benchmark case {index} sample {sample_index} is not an object")
            if expected_backend == "webgpu":
                if not isinstance(sample.get("gpu"), dict) or not sample["gpu"]:
                    raise ValueError(f"benchmark case {index} sample {sample_index} has no GPU timestamp profile")
                values = list(sample["gpu"].values())
                if any(not finite_number(value) or float(value) < 0 for value in values):
                    raise ValueError(f"benchmark case {index} sample {sample_index} contains an invalid GPU timing")
                totals.append(sum(float(value) for value in values))
            else:
                wall = sample.get("wallMs")
                if not finite_number(wall) or float(wall) <= 0:
                    raise ValueError(f"benchmark case {index} sample {sample_index} has no valid wall timing")
                totals.append(float(wall))
        median = statistics.median(totals)
        if not math.isfinite(median) or median <= 0:
            raise ValueError(f"benchmark case {index} has an invalid profiled median")
        medians.append(median)
    metric = math.exp(sum(math.log(value) for value in medians) / len(medians))
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("benchmark geometric mean is invalid")
    result["metricMs"] = metric
    result["metric"] = (
        "geometric mean of per-case median summed GPU kernel milliseconds"
        if expected_backend == "webgpu"
        else "geometric mean of per-case median wall-clock milliseconds"
    )
    result["profiledCaseMediansMs"] = medians
    return metric


def parity_metric(result: dict, max_dp: float, expected_backend: str = "webgpu") -> float:
    validate_backend(result, expected_backend)
    questions = result.get("questions")
    argmax = result.get("argmax")
    max_observed = result.get("maxDp")
    if not isinstance(questions, int) or not isinstance(argmax, int) or questions <= 0:
        raise ValueError("parity result has invalid question counts")
    if argmax != questions:
        raise ValueError(f"parity argmax mismatch: {argmax}/{questions}")
    if not finite_number(max_observed) or float(max_observed) > max_dp:
        raise ValueError(f"parity maxDp {max_observed!r} exceeds {max_dp}")
    result["maxDpLimit"] = max_dp
    return float(max_observed)


def latency_metric(result: dict, expected_backend: str = "webgpu") -> float:
    validate_backend(result, expected_backend)
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("latency result has no cases")
    medians: list[float] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not finite_number(case.get("p50")) or float(case["p50"]) <= 0:
            raise ValueError(f"latency case {index} has an invalid p50")
        medians.append(float(case["p50"]))
    metric = math.exp(sum(math.log(value) for value in medians) / len(medians))
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("latency geometric mean is invalid")
    result["metricMs"] = metric
    result["metric"] = "geometric mean of per-case wall-clock p50 milliseconds"
    result["p50CaseMediansMs"] = medians
    return metric


def manifest_file_entry(manifest: dict, path: str) -> dict | None:
    kevala = manifest.get("kevala")
    if isinstance(kevala, dict) and kevala.get("path") == path:
        return kevala
    semif = manifest.get("semif")
    fixtures = semif.get("fixtures") if isinstance(semif, dict) else None
    if isinstance(fixtures, list):
        for entry in fixtures:
            if isinstance(entry, dict) and entry.get("path") == path:
                return entry
    return None


def validate_decision_metadata(result: dict) -> None:
    """Validate provenance and timing controls emitted by the decision page."""
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("decision metadata is not an object")
    model_info = metadata.get("modelInfo")
    if not isinstance(model_info, dict):
        raise ValueError("decision metadata is missing the loaded model header")
    model_name = result.get("model")
    if not isinstance(model_name, str) or not model_name or model_info.get("name") != model_name:
        raise ValueError("decision result model does not match the loaded model header")
    for key in ("name", "revision", "source"):
        if not isinstance(model_info.get(key), str) or not model_info[key]:
            raise ValueError(f"decision model header is missing {key}")
    model_arch = metadata.get("modelArch")
    if not isinstance(model_arch, str) or not model_arch:
        raise ValueError("decision metadata is missing the loaded architecture")
    if result.get("modelInfo") != model_info or result.get("modelArch") != model_arch:
        raise ValueError("decision result and metadata model headers disagree")

    state_cache = metadata.get("stateCache")
    if not isinstance(state_cache, dict) or state_cache.get("enabled") is not False:
        raise ValueError(f"decision state cache must be disabled, got {state_cache!r}")
    if result.get("stateCache") != state_cache:
        raise ValueError("decision result and metadata state cache policies disagree")
    clock = metadata.get("clock")
    if not isinstance(clock, dict) or clock.get("quantized") is not False or clock.get("fine") is not True:
        raise ValueError(f"decision benchmark requires a fine, non-quantized clock, got {clock!r}")
    min_delta = clock.get("minDeltaMs")
    if not finite_number(min_delta) or float(min_delta) <= 0 or float(min_delta) >= 50:
        raise ValueError(f"decision clock has no fine resolution: {clock!r}")

    manifest = metadata.get("datasetManifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("version"), str) or not manifest["version"]:
        raise ValueError("decision metadata is missing the parsed dataset manifest")
    manifest_sha = metadata.get("datasetManifestSha256")
    if not isinstance(manifest_sha, str) or SHA256_RE.fullmatch(manifest_sha) is None:
        raise ValueError("decision metadata has an invalid dataset manifest SHA-256")
    files = metadata.get("datasetFiles")
    file_hashes = metadata.get("datasetFileHashes")
    if not isinstance(files, list) or not files or any(not isinstance(path, str) for path in files) or len(set(files)) != len(files):
        raise ValueError("decision metadata has no valid dataset file list")
    if not isinstance(file_hashes, dict):
        raise ValueError("decision metadata is missing dataset file hashes")
    if set(file_hashes) != set(files):
        raise ValueError("decision dataset file hashes do not match the selected files")
    for path in files:
        entry = manifest_file_entry(manifest, path)
        actual = file_hashes.get(path)
        if not isinstance(entry, dict) or not isinstance(actual, dict):
            raise ValueError(f"decision dataset provenance is missing {path}")
        if actual.get("rows") != entry.get("rows") or not isinstance(actual.get("rows"), int) or actual["rows"] <= 0:
            raise ValueError(f"decision dataset row count does not match the manifest for {path}")
        if actual.get("sha256") != entry.get("sha256") or SHA256_RE.fullmatch(str(actual.get("sha256"))) is None:
            raise ValueError(f"decision dataset SHA-256 does not match the manifest for {path}")


def validate_decision_probabilities(rows: list, probabilities: list) -> None:
    """Require every emitted probability map to match its typed option set exactly."""
    if len(probabilities) != len(rows):
        raise ValueError("decision rawProbs count does not match timing rows")
    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        options = row.get("options") if isinstance(row, dict) else None
        values = probability.get("probabilities") if isinstance(probability, dict) else None
        if not isinstance(probability, dict) or probability.get("caseId") != row.get("caseId") or probability.get("permutation") != row.get("permutation"):
            raise ValueError(f"decision probability row {index} has mismatched case/permutation identity")
        if not isinstance(options, list) or any(not isinstance(option, str) for option in options) or len(set(options)) != len(options):
            raise ValueError(f"decision timing row {index} has invalid option IDs")
        if not isinstance(values, dict) or set(values) != set(options) or len(values) != len(options):
            raise ValueError(f"decision probability row {index} has extra or missing option keys")
        numbers = [values[option] for option in options]
        if any(not finite_number(value) or float(value) < 0 or float(value) > 1 for value in numbers):
            raise ValueError(f"decision probability row {index} has an invalid value")
        if abs(sum(float(value) for value in numbers) - 1) > DECISION_PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"decision probability row {index} does not sum to one within {DECISION_PROBABILITY_SUM_TOLERANCE}")


def decision_metric(result: dict, expected_backend: str = "webgpu") -> float:
    """Validate one-model decision quality and return the independent p50 latency metric."""
    validate_backend(result, expected_backend)
    if result.get("status") != "done" or result.get("done") is not True:
        raise ValueError("decision benchmark did not finish")
    for key in ("model", "modelRevision", "hash", "hashScope", "catalogHash", "catalogHashScope", "backend", "gpuInfo", "timingAllRows", "rawProbs", "quality", "latency", "metadata"):
        if key not in result:
            raise ValueError(f"decision result is missing {key}")
    revision = result.get("modelRevision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("decision result has no model revision")
    pack_hash = result.get("hash")
    if not isinstance(pack_hash, str) or not pack_hash:
        raise ValueError("decision result has no catalog or header hash")
    catalog_hash = result.get("catalogHash")
    hash_scope = result.get("hashScope")
    if catalog_hash is not None:
        if SHA256_RE.fullmatch(str(catalog_hash)) is None or result.get("hash") != catalog_hash:
            raise ValueError("decision catalog hash is invalid or disagrees with hash")
        if hash_scope != "expected-catalog-pack-sha256" or result.get("catalogHashScope") != hash_scope:
            raise ValueError("decision catalog hash must be labeled as an expected catalog value")
    elif hash_scope != "pack-header-metadata-sha256" or result.get("catalogHashScope") is not None:
        raise ValueError("decision header hash has an invalid scope")
    validate_decision_metadata(result)
    errors = result.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError(f"decision benchmark reported errors: {errors!r}")
    quality = result["quality"]
    if not isinstance(quality, dict):
        raise ValueError("decision quality is not an object")
    total = quality.get("total")
    valid = quality.get("valid")
    correct = quality.get("correct")
    invalid = quality.get("invalidCount")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (total, valid, correct, invalid)) or total <= 0:
        raise ValueError(f"decision quality counts are invalid: {quality!r}")
    if valid != total or invalid != 0 or correct < 0 or correct > total:
        raise ValueError(f"decision quality has invalid rows: {quality!r}")
    if not finite_number(quality.get("accuracy")) or not math.isclose(
        float(quality["accuracy"]), correct / total, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"decision quality accuracy is inconsistent: {quality!r}")
    rows = result["timingAllRows"]
    if not isinstance(rows, list) or len(rows) != total:
        raise ValueError(f"decision timing rows {len(rows) if isinstance(rows, list) else rows!r} != quality total {total}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"decision timing row {index} is not an object")
        wall = row.get("wallMs")
        if not finite_number(wall) or float(wall) <= 0:
            raise ValueError(f"decision timing row {index} has invalid wallMs {wall!r}")
        if not isinstance(row.get("caseId"), str) or not isinstance(row.get("permutation"), (int, str)) or row.get("permutation") == "":
            raise ValueError(f"decision timing row {index} is missing its case/permutation identity")
    probs = result["rawProbs"]
    if not isinstance(probs, list) or len(probs) != total:
        raise ValueError("decision rawProbs count does not match quality total")
    validate_decision_probabilities(rows, probs)
    latency = result["latency"]
    if not isinstance(latency, dict):
        raise ValueError("decision latency is not an object")
    p50 = latency.get("p50Ms")
    p95 = latency.get("p95Ms")
    if not finite_number(p50) or not finite_number(p95) or float(p50) <= 0 or float(p95) <= 0 or float(p95) < float(p50):
        raise ValueError(f"decision latency percentiles are invalid: {latency!r}")
    if expected_backend == "webgpu":
        gpu = result["gpuInfo"]
        if not isinstance(gpu, dict) or gpu.get("available") is not True or gpu.get("isFallbackAdapter") is True:
            raise ValueError(f"decision benchmark did not report a hardware WebGPU adapter: {gpu!r}")
    result["metricMs"] = float(p50)
    result["metric"] = "per-decision wall-clock p50 milliseconds; quality validated independently"
    return float(p50)


def kernels_metric(result: dict) -> float:
    if result.get("done") is not True:
        raise ValueError("kernel test page did not finish")
    numeric = {key: value for key, value in result.items() if key not in {"done", "runner"}}
    if not numeric:
        raise ValueError("kernel test page returned no numeric errors")
    errors: list[float] = []
    for key, value in numeric.items():
        if not finite_number(value):
            raise ValueError(f"kernel result {key} is not finite")
        error = float(value)
        if error >= 1e-4:
            raise ValueError(f"kernel result {key} is {error:g}, expected < 1e-4")
        errors.append(error)
    result["maxError"] = max(errors)
    return max(errors)


def cache_metric(result: dict, expected_backend: str = "webgpu") -> float:
    validate_backend(result, expected_backend)
    limit = 0.0 if expected_backend == "webgpu" else 1e-5
    for key in ("hitVsMiss", "extVsFresh"):
        value = result.get(key)
        if not finite_number(value) or float(value) < 0 or float(value) > limit:
            requirement = "exactly zero" if limit == 0 else f"<= {limit:g}"
            raise ValueError(f"cache {key} must be {requirement}, got {value!r}")
    stats = result.get("stats")
    # CPU cache counters are checked by the native kev_cache integration test;
    # the browser's CPU ABI currently exposes only the probabilities and latency.
    if expected_backend == "webgpu" or stats is not None:
        if not isinstance(stats, dict) or any(stats.get(key) != 1 for key in ("hits", "misses", "extensions")):
            raise ValueError(f"cache check did not exercise a hit, miss and extension: {stats}")
    result["cacheCountersVerified"] = stats is not None
    return 0.0


def chromium_binary(explicit: str | None) -> str:
    """Find a Chromium executable without starting it or a browser session."""
    configured = explicit or next((os.environ.get(name) for name in ("CHROMIUM_BIN", "CHROME_BIN", "BROWSER_BIN") if os.environ.get(name)), None)
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise RuntimeError(f"Chromium executable is not executable: {path}")
    candidates: list[str] = []
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend(
        (
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome",
            "/opt/google/chrome/chrome",
        )
    )
    playwright_root = Path.home() / ".cache" / "ms-playwright"
    candidates.extend(str(path) for path in sorted(playwright_root.glob("chromium-*/chrome-linux*/chrome")))
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError("Chromium executable not found; set CHROMIUM_BIN or --binary")


def _chrome_diagnostic_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir() or directory.stat().st_uid != os.getuid():
        raise ValueError("Chrome diagnostics directory must be an owned, non-symlink directory")
    if any(directory.iterdir()):
        raise ValueError("Chrome diagnostics directory must be new and empty")
    return {"directory": directory, "chromeDriver": directory / "chromedriver.log", "chromeGpu": directory / "chrome-gpu.log"}


def _chrome_diagnostic_measure(paths: dict[str, Path]) -> dict:
    files = {}
    total = 0
    for name in ("chromeDriver", "chromeGpu"):
        path = paths[name]
        if path.is_symlink():
            raise ValueError(f"Chrome diagnostic {name} cannot be a symlink")
        if not path.exists():
            files[name] = {"path": str(path), "state": "missing", "bytes": 0}
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
            raise ValueError(f"Chrome diagnostic {name} must be an owned regular file")
        size = path.stat().st_size
        files[name] = {"path": str(path), "state": "present", "bytes": size}
        total += size
    return {"files": files, "totalBytes": total, "budgetBytes": CHROME_DIAGNOSTIC_BUDGET_BYTES,
        "overshootBytes": max(0, total - CHROME_DIAGNOSTIC_BUDGET_BYTES)}


def _q8_check_diagnostic_budget(paths: dict[str, Path], state: dict, stage: str) -> None:
    measured = _chrome_diagnostic_measure(paths)
    state["sampleCount"] += 1
    state["lastSampleStage"] = stage
    state["lastSample"] = measured
    state["peakBytes"] = max(state["peakBytes"], measured["totalBytes"])
    if measured["totalBytes"] > CHROME_DIAGNOSTIC_BUDGET_BYTES:
        state["budgetExceeded"] = {"stage": stage, **measured}
        raise ValueError(f"Chrome diagnostic log budget exceeded at {stage}; owned logs are retained with overshoot")


def precise_firefox_timers(args: argparse.Namespace) -> bool:
    return args.browser == "firefox" and not args.cdp and args.result in {"bench", "latency", "decision", "kernels"}


def create_browser(args: argparse.Namespace):
    """Create the requested Selenium browser after resolving its local binary."""
    if args.browser == "chromium":
        binary = chromium_binary(args.binary)
        options = ChromeOptions()
        options.binary_location = binary
        if args.result == "q8-swiglu":
            options.page_load_strategy = "none"
        options.add_argument("--headless=new") if not args.headed else None
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-dev-shm-usage")
        for argument in args.chromium_arg:
            options.add_argument(argument)
        diagnostics_dir = getattr(args, "chrome_diagnostics_dir", None)
        service_env = getattr(args, "chrome_service_env", None)
        if diagnostics_dir is not None:
            log_paths = getattr(args, "chrome_diagnostic_paths", None) or _chrome_diagnostic_paths(diagnostics_dir)
            options.add_argument("--enable-logging")
            options.add_argument("--log-level=1")
            service_env = dict(service_env or os.environ)
            service_env["CHROME_LOG_FILE"] = str(log_paths["chromeGpu"])
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            options.add_argument("--no-sandbox")
        if diagnostics_dir is not None:
            service = ChromeService(service_args=["--log-level=WARNING"], log_output=str(log_paths["chromeDriver"]), env=service_env)
        else:
            service = ChromeService(log_output=os.devnull, env=service_env)
        return webdriver.Chrome(options=options, service=service)

    binary = args.binary or os.environ.get("FIREFOX_BIN") or os.environ.get("BROWSER_BIN") or "/usr/bin/firefox"
    options = Options()
    options.binary_location = binary
    options.set_preference("dom.webgpu.enabled", True)
    options.set_preference("dom.webgpu.workers.enabled", True)
    # These preferences apply only to this benchmark's temporary Selenium profile.
    if precise_firefox_timers(args):
        options.set_preference("privacy.resistFingerprinting", False)
        options.set_preference("privacy.reduceTimerPrecision", False)
        options.set_preference("privacy.reduceTimerPrecision.jitter", False)
    options.add_argument("-no-remote")
    if not args.headed:
        options.add_argument("-headless")
    return webdriver.Firefox(options=options, service=Service(log_output=os.devnull))


def _q8_process(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 1 :].strip().split()
    cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    unified = next((line for line in cgroup if line.startswith("0::")), None)
    if unified is None:
        raise ValueError(f"PID {pid} has no unified cgroup identity")
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "startTicks": fields[19], "cgroup": unified}


def _q8_thread(pid: int, tid: int) -> dict:
    evidence: dict = {"pid": pid, "tid": tid, "phase": "before-read"}
    try:
        return _q8_thread_sample(pid, tid, evidence)
    except Exception as error:
        error.q8_thread_evidence = evidence
        raise


def _q8_thread_sample(pid: int, tid: int, evidence: dict) -> dict:
    directory = Path(f"/proc/{pid}/task/{tid}")
    evidence["phase"] = "before-stat-status-and-cgroup"
    raw = (directory / "stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 1 :].strip().split()
    task_id = int(raw[:raw.index("(")].strip())
    status = dict(line.split(":", 1) for line in (directory / "status").read_text(encoding="utf-8").splitlines() if ":" in line)
    cgroup = (directory / "cgroup").read_text(encoding="utf-8").splitlines()
    unified = next((line for line in cgroup if line.startswith("0::")), None)
    before = {"startTicks": fields[19], "state": fields[0]}
    evidence["before"] = {"taskId": task_id, **before, "tgid": int(status.get("Tgid", "-1")), "pid": int(status.get("Pid", "-1")), "cgroup": unified}
    if task_id != tid or int(status.get("Tgid", "-1")) != pid or int(status.get("Pid", "-1")) != tid or unified is None:
        raise ValueError(f"TID {tid} does not match process {pid} or has no unified cgroup identity")
    evidence["phase"] = "after-stat-status-and-cgroup"
    raw_after = (directory / "stat").read_text(encoding="utf-8")
    fields_after = raw_after[raw_after.rfind(")") + 1 :].strip().split()
    after_id = int(raw_after[:raw_after.index("(")].strip())
    cgroup_after = (directory / "cgroup").read_text(encoding="utf-8").splitlines()
    unified_after = next((line for line in cgroup_after if line.startswith("0::")), None)
    status_after = dict(line.split(":", 1) for line in (directory / "status").read_text(encoding="utf-8").splitlines() if ":" in line)
    evidence["after"] = {"taskId": after_id, "startTicks": fields_after[19], "state": fields_after[0],
        "tgid": int(status_after.get("Tgid", "-1")), "pid": int(status_after.get("Pid", "-1")), "cgroup": unified_after}
    if after_id != tid or fields_after[19] != before["startTicks"] or unified_after != unified:
        raise ValueError(f"TID {tid} changed identity or cgroup during its sample")
    if int(status_after.get("Tgid", "-1")) != pid or int(status_after.get("Pid", "-1")) != tid:
        raise ValueError(f"TID {tid} changed process ownership during its sample")
    return {"pid": pid, "tid": tid, "startTicks": before["startTicks"], "state": before["state"], "cgroup": unified, "cgroupAfter": unified_after}


def _q8_processes() -> dict[int, dict]:
    pids = sorted(int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit())
    if len(pids) > 32768:
        raise ValueError("Q8 SwiGLU process inventory exceeded its bound")
    rows = {}
    for pid in pids:
        try:
            rows[pid] = _q8_process(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            if Path(f"/proc/{pid}").stat().st_uid == os.getuid():
                raise ValueError("Q8 SwiGLU process inventory has an unreadable same-user process") from error
    return rows


def _q8_descendants(rows: dict[int, dict], roots: set[int]) -> set[int]:
    result = set(roots)
    while True:
        added = {pid for pid, row in rows.items() if row["ppid"] in result} - result
        if not added:
            return result
        result.update(added)


def _q8_app_scopes() -> list[dict]:
    args = ["/usr/bin/systemctl", "--user", "list-units", "--all", "--type=scope", "--plain", "--no-legend", "--no-pager", "app-org.chromium.Chromium-*.scope"]
    completed = subprocess.run(args, capture_output=True, timeout=5, check=False)
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024 or len(completed.stderr) > 65536:
        raise ValueError("Q8 SwiGLU Chromium app-scope query failed or exceeded its bound")
    result = []
    for line in completed.stdout.decode().splitlines():
        columns = line.split(maxsplit=4)
        if columns:
            if len(columns) < 4 or not columns[0].startswith("app-org.chromium.Chromium-") or not columns[0].endswith(".scope"):
                raise ValueError("Q8 SwiGLU app-scope query returned an unexpected row")
            result.append({"unit": columns[0], "load": columns[1], "active": columns[2], "sub": columns[3]})
    return result


def _q8_scope(unit: str) -> dict:
    args = ["/usr/bin/systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "ControlGroup", "-p", "MemoryHigh", "-p", "MemoryMax", "-p", "MemorySwapMax"]
    completed = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    if completed.returncode != 0:
        raise ValueError("Q8 SwiGLU systemd scope metadata query failed")
    properties = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    expected = {"MemoryHigh": str(32 * 1024**3), "MemoryMax": str(40 * 1024**3), "MemorySwapMax": str(4 * 1024**3)}
    if properties.get("ActiveState") != "active" or properties.get("SubState") != "running" or not properties.get("ControlGroup", "").endswith("/" + unit):
        raise ValueError("Q8 SwiGLU memory scope is not active with the expected identity")
    if any(properties.get(key) != value for key, value in expected.items()):
        raise ValueError("Q8 SwiGLU memory scope limits differ from the fixed 32/40/4 GiB protocol")
    cgroup = "0::" + properties["ControlGroup"]
    directory = Path("/sys/fs/cgroup") / properties["ControlGroup"].lstrip("/")
    for filename, expected_value in (("memory.high", expected["MemoryHigh"]), ("memory.max", expected["MemoryMax"]), ("memory.swap.max", expected["MemorySwapMax"])):
        if (directory / filename).read_text(encoding="ascii").strip() != expected_value:
            raise ValueError(f"Q8 SwiGLU cgroup {filename} differs from systemd scope")
    if _q8_process(os.getpid())["cgroup"] != cgroup:
        raise ValueError("Q8 SwiGLU controller is outside its declared cgroup")
    return {"unit": unit, "properties": properties, "cgroup": cgroup, "directory": str(directory), "limits": expected}


def _q8_bounded_snapshot_partial(partial: dict) -> dict:
    maximum = 512 * 1024
    def serialized(value: object) -> bytes:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()

    encoded = serialized(partial)
    if len(encoded) <= maximum:
        return partial

    def trim_rows(container: dict, key: str, keep: int) -> None:
        rows = container.get(key)
        if not isinstance(rows, list) or len(rows) <= keep:
            return
        full = serialized(rows)
        container[f"{key}Total"] = len(rows)
        container[f"{key}Sha256"] = hashlib.sha256(full).hexdigest()
        container[f"{key}Truncated"] = True
        container[key] = rows[:keep]

    for keep in (512, 256, 128, 64, 32, 16, 8, 4, 0):
        compact = json.loads(json.dumps(partial))
        phases = compact.get("phases", [])
        if len(phases) > 256:
            compact["phasesTotal"] = len(phases)
            compact["phasesSha256"] = hashlib.sha256(serialized(phases)).hexdigest()
            compact["phases"] = phases[-256:]
        for pass_record in compact.get("passes", []):
            for key in ("threads", "liveThreads", "threadsDisappearedDuringSample", "processes", "liveProcesses"):
                trim_rows(pass_record, key, min(keep, 64) if key in {"processes", "liveProcesses"} else keep)
            churn = pass_record.get("threadChurn")
            if isinstance(churn, dict):
                trim_rows(churn, "newThreads", keep)
                trim_rows(churn, "disappearedThreads", keep)
            pass_record["partialRowsTruncated"] = True
        encoded = serialized(compact)
        if len(encoded) <= maximum:
            return compact
    current = partial.get("current", {})
    minimal = {"stage": partial.get("stage"), "roots": partial.get("roots", [])[:256], "phases": partial.get("phases", [])[-32:],
        "current": current, "assurance": partial.get("assurance"), "error": str(partial.get("error", ""))[:4096], "truncated": True,
        "passes": []}
    for row in partial.get("passes", []):
        churn = row.get("threadChurn") or {}
        churn_summary = {}
        for key in ("newThreads", "disappearedThreads"):
            values = churn.get(key, []) if isinstance(churn, dict) else []
            churn_summary[key] = {"total": len(values), "sha256": hashlib.sha256(serialized(values)).hexdigest(), "sample": values[:4]}
        minimal["passes"].append({"scan": row.get("scan"), "membersBeforeCount": len(row.get("membersBefore", [])),
            "membersAfterCount": len(row.get("membersAfter", [])), "membersFinalCount": len(row.get("membersFinal", [])),
            "processCount": len(row.get("processes", [])), "threadCount": len(row.get("threads", [])),
            "liveThreadCount": len(row.get("liveThreads", [])), "churn": churn_summary})
    if len(serialized(minimal)) > maximum:
        minimal["current"] = {key: value for key, value in current.items() if key in {"phase", "pid", "tid", "error", "expected", "observed", "processBefore", "processAfter", "tidEvidence"}}
        minimal["phases"] = minimal["phases"][-8:]
        minimal["passes"] = minimal["passes"][:2]
        for row in minimal["passes"]:
            for item in row["churn"].values():
                item["sample"] = item["sample"][:1]
    if len(serialized(minimal)) > maximum:
        current_small = {key: value for key, value in minimal.get("current", {}).items() if key in {"phase", "pid", "tid", "error", "expected", "observed"}}
        current_small["error"] = str(current_small.get("error", ""))[:512]
        minimal["current"] = current_small
    if len(serialized(minimal)) > maximum:
        raise ValueError("Q8 SwiGLU partial snapshot cannot be bounded to 512 KiB")
    return minimal


def _q8_store_snapshot_failure(scope: dict, partial: dict) -> None:
    failures = scope.setdefault("snapshotFailures", [])
    if len(failures) < 2:
        candidate = [*failures, partial]
        if len(json.dumps(candidate, separators=(",", ":"), ensure_ascii=False).encode()) <= 1024 * 1024:
            failures.append(partial)
            return
    encoded = json.dumps(partial, separators=(",", ":"), ensure_ascii=False).encode()
    summary = {"stage": partial.get("stage"), "error": partial.get("error"), "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(), "storedFullPartial": False}
    if not failures:
        failures.append(summary)
    elif len(failures) == 1:
        failures.append(summary)
    else:
        failures[1] = summary


def _q8_snapshot_failure_reference(error: Exception, scope: dict | None) -> dict | None:
    partial = getattr(error, "q8_partial_snapshot", None)
    if not isinstance(partial, dict):
        return None
    encoded = json.dumps(partial, separators=(",", ":"), ensure_ascii=False).encode()
    reference = {"stage": partial.get("stage"), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
        "storedUnder": "containment.scope.snapshotFailures" if scope and partial in scope.get("snapshotFailures", []) else "exception-only"}
    return reference


def _q8_scope_snapshot(scope: dict, stage: str, roots: set[int], original_anchors: dict | None = None) -> dict:
    partial = {"stage": stage, "roots": sorted(roots), "phases": [], "passes": [],
        "assurance": "sequential per-process and per-TID identity/cgroup samples; not an atomic inventory or continuous containment"}

    def phase(name: str, **identity: object) -> None:
        partial["current"] = {"phase": name, **identity}
        partial["phases"].append({"phase": name, "status": "started"})

    try:
        directory = Path(scope["directory"])
        phase("read-members-before")
        members_before = set(map(int, (directory / "cgroup.procs").read_text(encoding="ascii").split()))
        processes = _q8_processes()
        owned = _q8_descendants(processes, roots) | members_before
        if len(owned) > 256:
            raise ValueError("Q8 SwiGLU owned browser process bound exceeded")
        pass_record = {"scan": 0, "membersBefore": sorted(members_before), "membersAfter": [], "membersFinal": [],
            "processes": [], "threads": [], "liveProcesses": [], "liveThreads": [], "threadsDisappearedDuringSample": []}
        partial["passes"].append(pass_record)
        verified_count = 0
        for pid in sorted(owned):
            phase("initial-process-bookend", pid=pid)
            row = processes.get(pid)
            if row is None:
                if pid in roots:
                    raise ValueError(f"Q8 SwiGLU required process PID {pid} disappeared at {stage}")
                continue
            process_row = {**row, "expected": row, "before": None, "after": None, "threadCount": 0}
            pass_record["processes"].append(process_row)
            partial["current"].update({"expectedProcess": row})
            process_before = _q8_process(pid)
            process_row["before"] = process_before
            process_row.update(process_before)
            partial["current"].update({"processBefore": process_before})
            if row["cgroup"] != scope["cgroup"]:
                raise ValueError(f"Q8 SwiGLU owned process PID {pid} escaped its exact cgroup")
            if (process_before["startTicks"], process_before["cgroup"]) != (row["startTicks"], scope["cgroup"]):
                raise ValueError(f"Q8 SwiGLU PID {pid} changed identity or cgroup during {stage}")
            first_thread_index = len(pass_record["threads"])
            phase("initial-tid-inventory", pid=pid, processStartTicks=process_before["startTicks"])
            for task in sorted(Path(f"/proc/{pid}/task").iterdir(), key=lambda path: int(path.name)):
                tid = int(task.name)
                partial["current"].update({"tid": tid})
                try:
                    thread = _q8_thread(pid, tid)
                except (FileNotFoundError, ProcessLookupError):
                    pass_record["threadsDisappearedDuringSample"].append({"pid": pid, "tid": tid, "phase": "initial-tid-inventory"})
                    continue
                except Exception as error:
                    partial["current"]["error"] = str(error)
                    if getattr(error, "q8_thread_evidence", None):
                        partial["current"]["tidEvidence"] = error.q8_thread_evidence
                    raise
                partial["current"]["tidEvidence"] = {key: thread.get(key) for key in ("pid", "tid", "startTicks", "cgroup", "cgroupAfter")}
                if thread["cgroup"] != scope["cgroup"] or thread["cgroupAfter"] != scope["cgroup"]:
                    raise ValueError(f"Q8 SwiGLU TID {tid} escaped its exact cgroup")
                thread["processStartTicks"] = process_before["startTicks"]
                pass_record["threads"].append(thread)
                verified_count += 1
                process_row["threadCount"] += 1
                if verified_count > 4096:
                    raise ValueError("Q8 SwiGLU owned thread inventory exceeded its bound")
            process_after = _q8_process(pid)
            process_row["after"] = process_after
            partial["current"].update({"processAfter": process_after})
            if (process_after["startTicks"], process_after["cgroup"]) != (process_before["startTicks"], scope["cgroup"]):
                raise ValueError(f"Q8 SwiGLU PID {pid} changed identity or cgroup during {stage}")
            for thread in pass_record["threads"][first_thread_index:]:
                thread["processStartTicksAfter"] = process_after["startTicks"]
        phase("process-cgroup-bookends")
        members_after = set(map(int, (directory / "cgroup.procs").read_text(encoding="ascii").split()))
        pass_record["membersAfter"] = sorted(members_after)
        processes_final = _q8_processes()
        if not members_after <= owned:
            partial["current"].update({"phase": "after-process-membership", "unexpectedMemberPids": sorted(members_after - owned),
                "unexpectedMemberProcesses": [processes_final[pid] for pid in sorted(members_after - owned) if pid in processes_final]})
            raise ValueError(f"Q8 SwiGLU cgroup gained unobserved processes at {stage}")
        descendants_final = _q8_descendants(processes_final, roots)
        if not descendants_final <= owned:
            partial["current"].update({"phase": "final-process-membership", "unexpectedDescendants": sorted(descendants_final - owned),
                "unexpectedDescendantProcesses": [processes_final[pid] for pid in sorted(descendants_final - owned) if pid in processes_final]})
            raise ValueError(f"Q8 SwiGLU browser ancestry gained unobserved processes during {stage}")
        for pid in sorted(descendants_final):
            first = processes.get(pid)
            current = processes_final.get(pid)
            phase("final-process-bookend", pid=pid)
            partial["current"].update({"expectedProcess": first, "observedProcess": current})
            if first is None or current is None or current["cgroup"] != scope["cgroup"] or current["startTicks"] != first["startTicks"]:
                raise ValueError(f"Q8 SwiGLU browser descendant PID {pid} escaped or changed identity during {stage}")
        members_final = set(map(int, (directory / "cgroup.procs").read_text(encoding="ascii").split()))
        pass_record["membersFinal"] = sorted(members_final)
        if not members_final <= owned:
            partial["current"].update({"phase": "final-cgroup-membership", "unexpectedMemberPids": sorted(members_final - owned),
                "unexpectedMemberProcesses": [processes_final[pid] for pid in sorted(members_final - owned) if pid in processes_final]})
            raise ValueError(f"Q8 SwiGLU cgroup gained unobserved processes during {stage}")
        for pid in sorted(members_final):
            first = processes.get(pid)
            current = processes_final.get(pid)
            phase("final-cgroup-member-bookend", pid=pid)
            partial["current"].update({"expectedProcess": first, "observedProcess": current})
            if first is None or current is None or current["cgroup"] != scope["cgroup"] or current["startTicks"] != first["startTicks"]:
                raise ValueError(f"Q8 SwiGLU cgroup member PID {pid} escaped or changed identity during {stage}")
        live_pids = _q8_descendants(processes_final, roots) | members_final
        if len(live_pids) > 256:
            raise ValueError("Q8 SwiGLU live browser process bound exceeded")
        live_count = 0
        for pid in sorted(live_pids):
            phase("current-process-bookend", pid=pid)
            expected = processes_final.get(pid)
            if expected is None:
                if pid in roots:
                    raise ValueError(f"Q8 SwiGLU required process PID {pid} disappeared during {stage}")
                continue
            partial["current"].update({"expectedProcess": expected})
            process_before = _q8_process(pid)
            live_index = len(pass_record["liveThreads"])
            live_process_row = {**expected, "expected": expected, "before": process_before, "after": None, "threadCount": 0}
            live_process_row.update(process_before)
            pass_record["liveProcesses"].append(live_process_row)
            partial["current"].update({"expectedProcess": expected, "processBefore": process_before})
            if expected["cgroup"] != scope["cgroup"]:
                raise ValueError(f"Q8 SwiGLU live process PID {pid} is outside its exact cgroup")
            if (process_before["startTicks"], process_before["cgroup"]) != (expected["startTicks"], scope["cgroup"]):
                raise ValueError(f"Q8 SwiGLU live PID {pid} changed identity or cgroup during {stage}")
            phase("current-tid-inventory", pid=pid, processStartTicks=process_before["startTicks"])
            for task in sorted(Path(f"/proc/{pid}/task").iterdir(), key=lambda path: int(path.name)):
                tid = int(task.name)
                partial["current"].update({"tid": tid})
                try:
                    thread = _q8_thread(pid, tid)
                except (FileNotFoundError, ProcessLookupError):
                    pass_record["threadsDisappearedDuringSample"].append({"pid": pid, "tid": tid, "phase": "current-tid-inventory"})
                    continue
                except Exception as error:
                    partial["current"]["error"] = str(error)
                    if getattr(error, "q8_thread_evidence", None):
                        partial["current"]["tidEvidence"] = error.q8_thread_evidence
                    raise
                partial["current"]["tidEvidence"] = {key: thread.get(key) for key in ("pid", "tid", "startTicks", "cgroup", "cgroupAfter")}
                if thread["cgroup"] != scope["cgroup"] or thread["cgroupAfter"] != scope["cgroup"]:
                    raise ValueError(f"Q8 SwiGLU live TID {tid} is outside its exact cgroup")
                thread["processStartTicks"] = process_before["startTicks"]
                pass_record["liveThreads"].append(thread)
                live_count += 1
                live_process_row["threadCount"] += 1
                if live_count > 4096:
                    raise ValueError("Q8 SwiGLU live thread inventory exceeded its bound")
            process_after = _q8_process(pid)
            live_process_row["after"] = process_after
            partial["current"].update({"processAfter": process_after})
            if (process_after["startTicks"], process_after["cgroup"]) != (process_before["startTicks"], scope["cgroup"]):
                raise ValueError(f"Q8 SwiGLU live PID {pid} changed identity or cgroup during {stage}")
            for thread in pass_record["liveThreads"][live_index:]:
                thread["processStartTicksAfter"] = process_after["startTicks"]
        phase("post-tid-process-bookends")
        members_post_threads = set(map(int, (directory / "cgroup.procs").read_text(encoding="ascii").split()))
        processes_post_threads = _q8_processes()
        descendants_post_threads = _q8_descendants(processes_post_threads, roots)
        pass_record["membersPostThreads"] = sorted(members_post_threads)
        partial["current"].update({"phase": "post-tid-process-membership", "expectedMembers": sorted(members_final),
            "observedMembers": sorted(members_post_threads), "expectedOwnedDescendants": sorted(owned),
            "observedDescendants": sorted(descendants_post_threads),
            "unexpectedProcesses": [processes_post_threads[pid] for pid in sorted((members_post_threads | descendants_post_threads) - owned) if pid in processes_post_threads]})
        if members_post_threads != members_final or not descendants_post_threads <= owned:
            raise ValueError(f"Q8 SwiGLU process membership changed during the current TID inventory at {stage}")
        for pid in sorted(descendants_post_threads | members_post_threads):
            before = processes_final.get(pid)
            current = processes_post_threads.get(pid)
            phase("post-tid-process-bookend", pid=pid)
            partial["current"].update({"expectedProcess": before, "observedProcess": current})
            if before is None or current is None or current["startTicks"] != before["startTicks"] or current["cgroup"] != scope["cgroup"]:
                raise ValueError(f"Q8 SwiGLU process PID {pid} changed identity or cgroup during the current TID inventory")
        pass_record["threadChurn"] = thread_identity_delta(pass_record["threads"], pass_record["liveThreads"], scope["cgroup"])
        pass_record["verifiedThreadCount"] = len(pass_record["threads"])
        pass_record["liveThreadCount"] = len(pass_record["liveThreads"])
        anchors = _q8_check_original_anchors(original_anchors or {}, _q8_process, phase)
        result = {"stage": stage, "roots": sorted(roots), "passes": [pass_record], "processes": pass_record["processes"],
            "membersBefore": pass_record["membersBefore"], "membersAfter": pass_record["membersAfter"], "membersFinal": pass_record["membersFinal"],
            "membersPostThreads": pass_record["membersPostThreads"], "threadCount": len(pass_record["liveThreads"]),
            "threadChurn": pass_record["threadChurn"], "anchorsAfterQuit": anchors,
            "assurance": "sequential individually verified process/TID samples with process identity and exact cgroup bookends; thread churn is retained; not atomic or continuous"}
        if len(json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()) > 2 * 1024**2:
            raise ValueError(f"Q8 SwiGLU process/TID evidence exceeded its 2 MiB bound at {stage}")
        partial["phases"].append({"phase": "complete", "status": "passed"})
        return result
    except Exception as error:
        partial["error"] = str(error)[:4096]
        if getattr(error, "q8_thread_evidence", None) and isinstance(partial.get("current"), dict):
            partial["current"]["tidEvidence"] = error.q8_thread_evidence
        if isinstance(partial.get("current"), dict) and isinstance(partial["current"].get("error"), str):
            partial["current"]["error"] = partial["current"]["error"][:4096]
        bounded = _q8_bounded_snapshot_partial(partial)
        error.q8_partial_snapshot = bounded
        _q8_store_snapshot_failure(scope, bounded)
        raise


def poll_chrome(cdp: str, url: str, expression: str, kind: str, timeout: float, headed: bool = False, default_context: bool = False) -> tuple[object, dict]:
    """Loads `url` in an owned target of a running Chrome and polls `expression` until it is done.

    This speaks the DevTools protocol directly: Playwright's attach step asserts on target types
    it does not know, which some Chrome builds add for their own UI.
    """
    import urllib.request

    from websockets.sync.client import connect

    endpoint = json.load(urllib.request.urlopen(f"{cdp}/json/version"))
    ids = iter(range(1, 1 << 30))

    with connect(endpoint["webSocketDebuggerUrl"], max_size=None, open_timeout=30) as ws:

        def send(method: str, params: dict | None = None, session: str | None = None) -> dict:
            msg_id = next(ids)
            message = {"id": msg_id, "method": method, "params": params or {}}
            if session:
                message["sessionId"] = session
            ws.send(json.dumps(message))
            while True:
                reply = json.loads(ws.recv(timeout=timeout))
                if reply.get("id") == msg_id:
                    if "error" in reply:
                        raise RuntimeError(f"{method}: {reply['error'].get('message')}")
                    return reply.get("result", {})

        context = None if default_context else send("Target.createBrowserContext", {"disposeOnDetach": True})["browserContextId"]
        target = None
        result: object = None
        try:
            target_params = {"url": "about:blank"}
            if context:
                target_params["browserContextId"] = context
            if headed:
                target_params["newWindow"] = True
            else:
                target_params.update({"hidden": True, "background": True})
            target = send("Target.createTarget", target_params)["targetId"]
            session = send("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
            if headed:
                window = send("Browser.getWindowForTarget", {"targetId": target})["windowId"]
                send("Browser.setWindowBounds", {"windowId": window, "bounds": {"width": 1280, "height": 900, "windowState": "normal"}})
                send("Page.bringToFront", session=session)
            send("Page.navigate", {"url": url}, session=session)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    evaluated = send("Runtime.evaluate", {"expression": f"(() => {{ {expression} }})()", "returnByValue": True}, session=session)
                except RuntimeError:  # no execution context yet, while the page navigates
                    time.sleep(0.25)
                    continue
                result = evaluated.get("result", {}).get("value")
                if result_is_done(result, kind):
                    break
                time.sleep(0.25)
        except Exception as error:
            result = {"error": f"browser runner: {error}"}
        finally:
            cleanup = []
            if target:
                cleanup.append(("Target.closeTarget", {"targetId": target}))
            if context:
                cleanup.append(("Target.disposeBrowserContext", {"browserContextId": context}))
            for method, params in cleanup:
                try:
                    send(method, params)
                except Exception as error:
                    print(f"browser cleanup: {error}", file=sys.stderr)
    return result, {"browserName": "chrome", "browserVersion": endpoint.get("Browser"), "browserContext": "default" if default_context else "isolated"}


def main() -> int:
    args = parse_args()
    private_bus = None
    q8_scope = None
    q8_containment = None
    q8_anchors = {}
    chrome_diagnostic_paths = None
    chrome_diagnostic_state = None
    if args.result == "q8-swiglu":
        if args.browser != "chromium" or args.backend != "webgpu" or args.cdp or not args.q8_swiglu_session_bus_helper or not args.q8_swiglu_scope_unit or not args.chrome_diagnostics_dir:
            print("q8-swiglu requires fresh Chromium, WebGPU, a private session-bus helper, owned systemd scope, and owned diagnostics directory", file=sys.stderr)
            return 2
        spec = importlib.util.spec_from_file_location("kevala_q8_swiglu_session_bus", args.q8_swiglu_session_bus_helper)
        if spec is None or spec.loader is None:
            print("q8-swiglu could not load its reviewed session-bus helper", file=sys.stderr)
            return 2
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        q8_containment = {"sessionBus": {}, "scope": None, "snapshots": [], "appScopesBefore": None, "appScopesAfter": None, "newAppScopes": None,
            "diagnosticLogs": None, "assurance": "bounded process/TID identity and exact cgroup checks at bookends; no continuous fork/exec or CPU-affinity claim"}
        try:
            private_bus = helper.PrivateSessionBus(q8_containment["sessionBus"])
            private_bus.check("before-browser-create")
            args.chrome_service_env = private_bus.service_environment(dict(os.environ))
            chrome_diagnostic_paths = _chrome_diagnostic_paths(args.chrome_diagnostics_dir)
            args.chrome_diagnostic_paths = chrome_diagnostic_paths
            chrome_diagnostic_state = {"directory": str(chrome_diagnostic_paths["directory"]), "budgetBytes": CHROME_DIAGNOSTIC_BUDGET_BYTES,
                "sampleIntervalMs": 250, "sampleCount": 0, "lastSampleStage": None, "lastSample": None, "peakBytes": 0, "budgetExceeded": None,
                "retained": True, "level": {"chromeDriver": "WARNING", "chromeGpu": "WARNING"}, "files": {"chromeDriver": "chromedriver.log", "chromeGpu": "chrome-gpu.log"}}
            q8_containment["diagnosticLogs"] = chrome_diagnostic_state
            q8_scope = _q8_scope(args.q8_swiglu_scope_unit)
            q8_containment["scope"] = q8_scope
            q8_containment["appScopesBefore"] = _q8_app_scopes()
            q8_containment["snapshots"].append(_q8_scope_snapshot(q8_scope, "before-browser-create", {os.getpid()}))
        except Exception as error:
            failure_reference = _q8_snapshot_failure_reference(error, q8_scope)
            if failure_reference:
                q8_containment["snapshotFailureReference"] = failure_reference
                if failure_reference["storedUnder"] == "exception-only":
                    q8_containment["snapshotFailure"] = error.q8_partial_snapshot
            print(f"q8-swiglu containment admission failed: {error}", file=sys.stderr)
            if private_bus is not None:
                try:
                    private_bus.close()
                except Exception as cleanup_error:
                    print(f"q8-swiglu session-bus cleanup failed: {cleanup_error}", file=sys.stderr)
            if args.output:
                failure_result = {"error": f"q8-swiglu containment admission failed: {error}", "runner": {"containment": q8_containment}}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(failure_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 2
    elif args.q8_swiglu_session_bus_helper or args.q8_swiglu_scope_unit:
        print("Q8 SwiGLU containment arguments require --result=q8-swiglu", file=sys.stderr)
        return 2
    if args.cdp_default_context and not args.cdp:
        print("--cdp-default-context requires --cdp", file=sys.stderr)
        return 2
    if args.result in {"gpu", "kernels"} and args.backend != "webgpu":
        print(f"--backend {args.backend} is not valid for --result {args.result}", file=sys.stderr)
        return 2
    url = args.url or (
        url_with_backend(DEFAULT_URLS[args.result], args.backend)
        if args.result in {"bench", "latency", "parity", "decision", "cache", "tetris"}
        else DEFAULT_URLS[args.result]
    )
    browser = None
    capabilities: dict = {}
    result: object = None
    result_global = RESULT_GLOBALS[args.result]
    result_expression = f"return window.{result_global} || null;"
    if args.cdp:
        result, capabilities = poll_chrome(args.cdp, url, result_expression, args.result, args.timeout, headed=args.headed, default_context=args.cdp_default_context)
    else:
        try:
            browser = create_browser(args)
            browser.set_page_load_timeout(args.timeout)
            browser.set_script_timeout(args.timeout)
            capabilities = dict(browser.capabilities)
            if args.result == "q8-swiglu":
                _q8_check_diagnostic_budget(chrome_diagnostic_paths, chrome_diagnostic_state, "after-browser-create")
                driver_pid = browser.service.process.pid
                browser_pid = int(browser.capabilities["goog:processID"])
                q8_anchors = {"driver": _q8_process(driver_pid), "browser": _q8_process(browser_pid)}
                q8_containment["snapshots"].append(_q8_scope_snapshot(q8_scope, "after-browser-create", {os.getpid(), driver_pid, browser_pid}))
            deadline = time.monotonic() + args.timeout
            if args.result == "q8-swiglu": q8_containment["snapshots"].append(_q8_scope_snapshot(q8_scope, "before-navigation", {os.getpid(), driver_pid, browser_pid}))
            browser.get(url)
            if args.result == "q8-swiglu":
                q8_terminal, result = poll_q8_page(
                    browser.execute_script,
                    deadline,
                    lambda stage: _q8_check_diagnostic_budget(chrome_diagnostic_paths, chrome_diagnostic_state, stage),
                )
            else:
                while time.monotonic() < deadline:
                    result = browser.execute_script(result_expression)
                    if result_is_done(result, args.result):
                        break
                    time.sleep(0.25)
            if args.result == "q8-swiglu":
                if not q8_terminal:
                    result = {"error": "q8-swiglu page did not expose its terminal flag before timeout"}
                result_snapshot = _q8_scope_snapshot(q8_scope, "after-benchmark-result", {os.getpid(), driver_pid, browser_pid})
                q8_containment["snapshots"].append(result_snapshot)
                process_rows = {row["pid"]: row for row in result_snapshot["processes"]}
                browser_tree = _q8_descendants(process_rows, {driver_pid, browser_pid})
                q8_anchors = {f"browser-driver-pid-{pid}": process_rows[pid] for pid in sorted(browser_tree) if pid in process_rows}
        except Exception as error:
            failure_reference = _q8_snapshot_failure_reference(error, q8_scope)
            if failure_reference:
                q8_containment["snapshotFailureReference"] = failure_reference
                if failure_reference["storedUnder"] == "exception-only":
                    q8_containment["snapshotFailure"] = error.q8_partial_snapshot
            result = {"error": f"browser runner: {error}"}
        finally:
            if browser is not None:
                try:
                    browser.quit()
                except Exception as error:
                    if isinstance(result, dict): result["error"] = f"browser cleanup: {error}"
                if args.result == "q8-swiglu":
                    try:
                        q8_containment["snapshots"].append(_q8_scope_snapshot(q8_scope, "after-browser-quit", {os.getpid()}, q8_anchors))
                        _q8_check_diagnostic_budget(chrome_diagnostic_paths, chrome_diagnostic_state, "after-browser-quit")
                        q8_containment["appScopesAfter"] = _q8_app_scopes()
                        before_units = {row["unit"] for row in q8_containment["appScopesBefore"]}
                        q8_containment["newAppScopes"] = [row for row in q8_containment["appScopesAfter"] if row["unit"] not in before_units]
                        if q8_containment["newAppScopes"]:
                            raise ValueError("Chromium created a new app scope outside the recorded benchmark cgroup")
                    except Exception as error:
                        q8_containment["error"] = str(error)
                if private_bus is not None:
                    try:
                        private_bus.close()
                    except Exception as error:
                        q8_containment["sessionBusError"] = str(error)
            elif private_bus is not None:
                try:
                    private_bus.close()
                except Exception as error:
                    q8_containment["sessionBusError"] = str(error)
            if args.result == "q8-swiglu" and browser is None:
                try:
                    _q8_check_diagnostic_budget(chrome_diagnostic_paths, chrome_diagnostic_state, "browser-not-created")
                except Exception as error:
                    q8_containment["diagnosticError"] = str(error)

    if not isinstance(result, dict) or not result_is_done(result, args.result):
        result = {"error": f"{args.result} page did not expose a completed result before timeout"}
    runner = {
        "browser": capabilities.get("browserName"),
        "browserRequested": "chrome (cdp)" if args.cdp else args.browser,
        "browserVersion": capabilities.get("browserVersion"),
        "headless": not args.headed and not args.cdp,
        "targetVisibility": "hidden" if args.cdp and not args.headed else "headed" if args.headed else "headless",
        "browserContext": capabilities.get("browserContext", "isolated"),
        "chromiumArgs": args.chromium_arg if not args.cdp and args.browser == "chromium" else [],
        "expectedBackend": args.backend,
        "vkDriverFiles": os.environ.get("VK_DRIVER_FILES"),
        "timerPrivacy": {
            "isolatedSeleniumProfile": not bool(args.cdp),
            "firefoxPrecisionDisabled": precise_firefox_timers(args),
            "note": "Firefox timer privacy settings are changed only in the temporary Selenium profile" if precise_firefox_timers(args) else "unchanged",
        },
    }
    if args.result == "q8-swiglu":
        if chrome_diagnostic_state is not None:
            q8_containment["diagnosticLogs"] = chrome_diagnostic_state
        runner["containment"] = q8_containment
    result["runner"] = runner
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        return 1
    actual_threads: int | None = None
    try:
        if args.result == "gpu":
            actual_threads = validate_backend(result, "webgpu")
            if result.get("status") != "done":
                raise ValueError("GPU benchmark did not finish")
            adapter = result.get("adapter")
            if not isinstance(adapter, dict) or adapter.get("isFallbackAdapter"):
                raise ValueError("benchmark did not use a hardware WebGPU adapter")
            correctness = result.get("correctness")
            if not isinstance(correctness, dict) or correctness.get("ok") is not True:
                raise ValueError("GPU benchmark correctness checks failed")
            metric = result.get("metricMs")
            if not finite_number(metric) or float(metric) <= 0:
                raise ValueError(f"invalid metricMs: {metric!r}")
        elif args.result == "bench":
            actual_threads = validate_backend(result, args.backend)
            metric = bench_metric(result, args.backend)
        elif args.result == "latency":
            actual_threads = validate_backend(result, args.backend)
            metric = latency_metric(result, args.backend)
        elif args.result == "parity":
            actual_threads = validate_backend(result, args.backend)
            metric = parity_metric(result, args.max_dp, args.backend)
        elif args.result == "cache":
            actual_threads = validate_backend(result, args.backend)
            metric = cache_metric(result, args.backend)
        elif args.result == "decision":
            actual_threads = validate_backend(result, args.backend)
            metric = decision_metric(result, args.backend)
        elif args.result == "q8-swiglu":
            actual_threads = validate_backend(result, args.backend)
            trial = result.get("q8SwiGluTrial")
            if not isinstance(trial, dict) or trial.get("status") != "complete" or len(trial.get("aa", [])) != 9 or len(trial.get("paired", [])) != 9:
                raise ValueError("Q8 SwiGLU result is incomplete")
            fixture = trial.get("fixtureIdentity")
            layers = fixture.get("plannedLayers") if isinstance(fixture, dict) else None
            if not isinstance(layers, list) or not layers or len(set(layers)) != len(layers) or not isinstance(fixture.get("requestHash"), str):
                raise ValueError("Q8 SwiGLU fixture permit identity is missing or invalid")
            if not isinstance(fixture.get("T1"), int) or fixture["T1"] < 1 or not isinstance(fixture.get("T2"), int) or fixture["T2"] < 1:
                raise ValueError("Q8 SwiGLU fixture permit stage shape is missing or invalid")
            verification = trial.get("verification", {}).get("trial", {})
            if verification.get("activationExact") is not True or verification.get("permitState") != "consumed" or verification.get("effectiveMode") != "verify":
                raise ValueError("Q8 SwiGLU activation parity did not pass")
            probe = trial.get("shapeProbe", {})
            if probe.get("requestHash") != fixture["requestHash"] or probe.get("trial", {}).get("effectiveMode") != "off":
                raise ValueError("Q8 SwiGLU off-shape probe does not bind the fixture permit")
            expected_checks = [check for check in verification.get("activationChecks", []) if check.get("layer") in layers]
            if len(expected_checks) != len(layers) or any(check.get("comparedElements") != fixture["T1"] * 12288 or check.get("guardComparisons") != 128 for check in expected_checks):
                raise ValueError("Q8 SwiGLU activation comparison counts are incomplete")
            if verification.get("expectedActivationLayers") != layers or verification.get("verifiedActivationLayers") != layers or verification.get("unexecutedActivationLayers") != probe.get("trial", {}).get("unexecutedActivationLayers"):
                raise ValueError("Q8 SwiGLU planned and verified prefill layer sets differ")
            checks = verification.get("activationChecks", [])
            unexecuted = verification.get("unexecutedActivationLayers", [])
            if not isinstance(checks, list) or not isinstance(unexecuted, list) or any(not isinstance(check, dict) for check in checks) or any(not isinstance(layer, int) or layer < 0 or layer >= len(checks) for layer in unexecuted):
                raise ValueError("Q8 SwiGLU unexecuted prefill layer metadata is invalid")
            if any(checks[layer].get("status") != "not-planned-in-prefill" for layer in unexecuted):
                raise ValueError("Q8 SwiGLU unexecuted prefill layers were reported as compared")
            candidates = [row.get("trial", {}) for row in trial.get("warmupsByMode", {}).get("candidate", [])]
            candidates.extend(pair.get("candidate", {}).get("trial", {}) for pair in trial.get("paired", []))
            if len(candidates) != 12 or any(stats.get("effectiveMode") != "candidate" or stats.get("permitState") != "consumed" or
                    stats.get("fusedDispatches") != len(layers) or stats.get("skippedSilumul") != len(layers) or
                    stats.get("fusedActivationLayers") != layers for stats in candidates):
                raise ValueError("Q8 SwiGLU timed candidate lanes did not consume their exact one-shot layer permits")
            if any(row.get("requestHash") != fixture["requestHash"] or row.get("trial", {}).get("stage1Tokens") != fixture["T1"] or
                    row.get("trial", {}).get("stage2Tokens") != fixture["T2"] for row in trial.get("warmupsByMode", {}).get("candidate", [])):
                raise ValueError("Q8 SwiGLU candidate warmups changed the latched long-request shape")
            if any(pair.get("candidate", {}).get("requestHash") != fixture["requestHash"] or pair.get("baseline", {}).get("requestHash") != fixture["requestHash"] for pair in trial.get("paired", [])):
                raise ValueError("Q8 SwiGLU paired lanes changed the latched canonical request hash")
            medium = trial.get("mediumFallback", {})
            medium_trial = medium.get("candidateMode", {}).get("trial", {})
            if medium.get("candidateMode", {}).get("requestHash") == fixture["requestHash"] or medium.get("outputCheck", {}).get("exact") is not True or medium_trial.get("effectiveMode") != "off" or medium_trial.get("permitState") != "absent" or medium_trial.get("fusedDispatches") != 0:
                raise ValueError("Q8 SwiGLU medium fallback did not remain exact")
            if trial.get("acceptance", {}).get("promote") is not False:
                raise ValueError("Q8 SwiGLU trial must not promote a production default")
            containment = result.get("runner", {}).get("containment")
            if not isinstance(containment, dict) or containment.get("error") or containment.get("sessionBusError") or containment.get("newAppScopes"):
                raise ValueError("Q8 SwiGLU browser containment evidence is incomplete")
            diagnostics = containment.get("diagnosticLogs")
            log_files = diagnostics.get("lastSample", {}).get("files", {}) if isinstance(diagnostics, dict) else {}
            if (not isinstance(diagnostics, dict) or diagnostics.get("budgetExceeded") or diagnostics.get("budgetBytes") != CHROME_DIAGNOSTIC_BUDGET_BYTES or
                    diagnostics.get("level") != {"chromeDriver": "WARNING", "chromeGpu": "WARNING"} or
                    set(log_files) != {"chromeDriver", "chromeGpu"} or any(row.get("state") != "present" for row in log_files.values()) or
                    diagnostics.get("lastSample", {}).get("totalBytes", CHROME_DIAGNOSTIC_BUDGET_BYTES + 1) > CHROME_DIAGNOSTIC_BUDGET_BYTES):
                raise ValueError("Q8 SwiGLU retained Chrome diagnostic logs are missing or exceeded their fixed budget")
            bus = containment.get("sessionBus", {})
            if bus.get("cleanup", {}).get("removed") is not True or bus.get("parentEnvironmentBefore") != bus.get("parentEnvironmentAfter") or not bus.get("checks") or any(check.get("status") != "passed" for check in bus["checks"]):
                raise ValueError("Q8 SwiGLU private session-bus endpoint evidence did not pass")
            stages = [row.get("stage") for row in containment.get("snapshots", [])]
            if stages != ["before-browser-create", "after-browser-create", "before-navigation", "after-benchmark-result", "after-browser-quit"]:
                raise ValueError("Q8 SwiGLU owned cgroup bookends are incomplete")
            metric = statistics.median(pair["candidate"]["wallMs"] for pair in trial["paired"])
        elif args.result == "tetris":
            summary = result["summary"]
            games = result["games"]
            if not games or sum(game.get("moves", 0) for game in games) != summary.get("moves"):
                raise ValueError("Tetris did not report consistent completed games")
            metric = summary.get("p50TotalMs")
            if summary.get("policy") != "drop":
                actual_threads = validate_backend(summary, args.backend)
                if not finite_number(metric) or metric <= 0 or not summary.get("selectedPlacements"):
                    raise ValueError("Tetris did not complete model decisions")
            elif metric != 0:
                raise ValueError("drop baseline unexpectedly reports model timing")
        else:
            metric = kernels_metric(result)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if actual_threads is not None:
        runner["actualThreads"] = actual_threads
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.result in {"gpu", "bench", "latency", "decision"}:
        print(json.dumps({"backend": result.get("backend"), "method": result.get("method") or result.get("metric"), "metricMs": metric}, sort_keys=True), file=sys.stderr)
    elif args.result == "parity":
        print(json.dumps({"backend": result.get("backend"), "argmax": result.get("argmax"), "questions": result.get("questions"), "maxDp": metric}, sort_keys=True), file=sys.stderr)
    elif args.result == "tetris":
        print(json.dumps({key: result["summary"].get(key) for key in ("model", "backend", "moves", "lines", "meanHoles", "p50TotalMs")}), file=sys.stderr)
    else:
        print(json.dumps({"maxError": metric}, sort_keys=True), file=sys.stderr)
    print(f"{metric:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
