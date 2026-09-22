#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "selenium>=4.25,<5",
# ]
# ///
"""Run a deterministic browser benchmark in an isolated headless browser.

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        choices=("gpu", "bench", "latency", "parity", "kernels", "cache"),
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
    return parser.parse_args()


DEFAULT_URLS = {
    "gpu": "http://127.0.0.1:18086/dev/gpu-bench.html",
    "bench": "http://127.0.0.1:18086/bench.html#auto&backend=webgpu&pack=local&model=laya&profile=1&runs=10&unique=1",
    "latency": "http://127.0.0.1:18086/bench.html#auto&backend=webgpu&pack=local&model=laya&profile=0&runs=10&unique=1",
    "parity": "http://127.0.0.1:18086/parity.html#auto&backend=webgpu&pack=local",
    "kernels": "http://127.0.0.1:18086/dev/kernels.html",
    "cache": "http://127.0.0.1:18086/dev/cache-test.html?backend=webgpu",
}


RESULT_GLOBALS = {
    "gpu": "gpuBench",
    "bench": "bench",
    "latency": "bench",
    "parity": "parity",
    "kernels": "kernelResults",
    "cache": "ct",
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
    field = "query"
    source = getattr(parts, field)
    params = parse_qsl(source, keep_blank_values=True)
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
    if field == "fragment":
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, updated))
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
    options.add_argument("-no-remote")
    if not args.headed:
        options.add_argument("-headless")
    return webdriver.Firefox(options=options, service=Service(log_output=os.devnull))


def main() -> int:
    args = parse_args()
    if args.result in {"gpu", "kernels"} and args.backend != "webgpu":
        print(f"--backend {args.backend} is not valid for --result {args.result}", file=sys.stderr)
        return 2
    url = args.url or (
        url_with_backend(DEFAULT_URLS[args.result], args.backend)
        if args.result in {"bench", "latency", "parity", "cache"}
        else DEFAULT_URLS[args.result]
    )
    browser = None
    capabilities: dict = {}
    result: object = None
    result_global = RESULT_GLOBALS[args.result]
    result_expression = f"return window.{result_global} || null;"
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
        "browserRequested": args.browser,
        "browserVersion": capabilities.get("browserVersion"),
        "headless": not args.headed,
        "expectedBackend": args.backend,
        "vkDriverFiles": os.environ.get("VK_DRIVER_FILES"),
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
        else:
            metric = kernels_metric(result)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if actual_threads is not None:
        runner["actualThreads"] = actual_threads
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.result in {"gpu", "bench", "latency"}:
        print(json.dumps({"backend": result.get("backend"), "method": result.get("method") or result.get("metric"), "metricMs": metric}, sort_keys=True), file=sys.stderr)
    elif args.result == "parity":
        print(json.dumps({"backend": result.get("backend"), "argmax": result.get("argmax"), "questions": result.get("questions"), "maxDp": metric}, sort_keys=True), file=sys.stderr)
    else:
        print(json.dumps({"maxError": metric}, sort_keys=True), file=sys.stderr)
    print(f"{metric:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
