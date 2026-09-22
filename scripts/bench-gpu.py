#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "selenium>=4.25,<5",
#   "websockets>=13",
# ]
# ///
"""Run a deterministic browser benchmark in an isolated headless browser, or in a running Chrome (--cdp).

The page owns timing and correctness checks. This wrapper only controls the
browser, validates the requested backend, saves the structured result, and
prints the scalar metric last so it can be consumed by an autoresearch loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        choices=("gpu", "bench", "latency", "parity", "decision", "kernels", "cache", "tetris"),
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
    parser.add_argument("--headed", action="store_true", help="show the selected browser instead of using headless mode")
    parser.add_argument(
        "--cdp",
        default=os.environ.get("KEVALA_BENCH_CDP"),
        help="run in the Chrome listening at this DevTools URL (a fresh context of it) instead of launching a browser",
    )
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
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return True
    if "done" in result:
        return result.get("done") is True
    if "status" in result:
        return result.get("status") in {"done", "error"}
    if kind == "gpu":
        return result.get("status") in {"done", "error"}
    if kind == "kernels":
        return result.get("done") is True
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
    if reported is not None and reported != actual_threads:
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
    if not finite_number(quality.get("accuracy")) or float(quality["accuracy"]) != correct / total:
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


def create_browser(args: argparse.Namespace):
    """Create the requested Selenium browser after resolving its local binary."""
    if args.browser == "chromium":
        binary = chromium_binary(args.binary)
        options = ChromeOptions()
        options.binary_location = binary
        options.add_argument("--headless=new") if not args.headed else None
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-dev-shm-usage")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            options.add_argument("--no-sandbox")
        return webdriver.Chrome(options=options, service=ChromeService(log_output=os.devnull))

    binary = args.binary or os.environ.get("FIREFOX_BIN") or os.environ.get("BROWSER_BIN") or "/usr/bin/firefox"
    options = Options()
    options.binary_location = binary
    options.set_preference("dom.webgpu.enabled", True)
    options.set_preference("dom.webgpu.workers.enabled", True)
    # Selenium creates a fresh temporary profile for this process. Relax Firefox's timer privacy
    # settings only for the isolated decision benchmark profile, never for a user's browser.
    if args.result == "decision":
        options.set_preference("privacy.resistFingerprinting", False)
        options.set_preference("privacy.reduceTimerPrecision", False)
        options.set_preference("privacy.reduceTimerPrecision.jitter", False)
    options.add_argument("-no-remote")
    if not args.headed:
        options.add_argument("-headless")
    return webdriver.Firefox(options=options, service=Service(log_output=os.devnull))


def poll_chrome(cdp: str, url: str, expression: str, kind: str, timeout: float) -> tuple[object, dict]:
    """Loads `url` in a fresh context of a running Chrome and polls `expression` until it is done.

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

        context = send("Target.createBrowserContext", {"disposeOnDetach": True})["browserContextId"]
        result: object = None
        try:
            target = send("Target.createTarget", {"url": "about:blank", "browserContextId": context, "newWindow": True})["targetId"]
            session = send("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
            # Chrome slows covered windows: size this one and bring it to the front
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
            send("Target.disposeBrowserContext", {"browserContextId": context})
    return result, {"browserName": "chrome", "browserVersion": endpoint.get("Browser")}


def main() -> int:
    args = parse_args()
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
        result, capabilities = poll_chrome(args.cdp, url, result_expression, args.result, args.timeout)
    else:
        try:
            browser = create_browser(args)
            browser.set_page_load_timeout(args.timeout)
            browser.set_script_timeout(args.timeout)
            capabilities = dict(browser.capabilities)
            deadline = time.monotonic() + args.timeout
            browser.get(url)
            while time.monotonic() < deadline:
                result = browser.execute_script(result_expression)
                if result_is_done(result, args.result):
                    break
                time.sleep(0.25)
        except Exception as error:
            result = {"error": f"browser runner: {error}"}
        finally:
            if browser is not None:
                browser.quit()

    if not isinstance(result, dict) or not result_is_done(result, args.result):
        result = {"error": f"{args.result} page did not expose a completed result before timeout"}
    runner = {
        "browser": capabilities.get("browserName"),
        "browserRequested": "chrome (cdp)" if args.cdp else args.browser,
        "browserVersion": capabilities.get("browserVersion"),
        "headless": not args.headed and not args.cdp,
        "expectedBackend": args.backend,
        "vkDriverFiles": os.environ.get("VK_DRIVER_FILES"),
        "timerPrivacy": {
            "isolatedSeleniumProfile": not bool(args.cdp),
            "firefoxPrecisionDisabled": args.result == "decision" and args.browser == "firefox" and not bool(args.cdp),
            "note": "Firefox timer privacy settings are changed only in the temporary Selenium profile" if args.result == "decision" and args.browser == "firefox" and not args.cdp else "unchanged",
        },
    }
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
