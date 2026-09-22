#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "playwright>=1.50",
#   "selenium>=4.25,<5",
# ]
# ///
"""Run the deterministic WebGPU benchmark in an isolated headless Firefox, or in a running Chrome.

The page owns GPU timing and correctness checks. This wrapper only controls the
browser, saves the structured result, and prints the scalar metric last so it
can be consumed by an autoresearch loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from selenium import webdriver
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
    parser.add_argument("--timeout", type=float, default=240.0, help="maximum browser wait in seconds")
    parser.add_argument("--output", type=Path, help="write the structured page result to this JSON file")
    parser.add_argument("--max-dp", type=float, default=0.024, help="maximum parity probability difference")
    parser.add_argument("--headed", action="store_true", help="show Firefox instead of using headless mode")
    parser.add_argument(
        "--cdp",
        default=os.environ.get("KEVALA_BENCH_CDP"),
        help="run in the Chrome listening at this DevTools URL (a fresh context of it) instead of Firefox",
    )
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


def result_is_done(result: object, kind: str) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return True
    if kind == "gpu":
        return result.get("status") in {"done", "error"}
    if kind == "kernels":
        return result.get("done") is True
    return isinstance(result.get("backend"), str)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def bench_metric(result: dict) -> float:
    if result.get("backend") != "webgpu":
        raise ValueError(f"benchmark backend is {result.get('backend')!r}, expected 'webgpu'")
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark has no cases")
    medians: list[float] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("samples"), list) or not case["samples"]:
            raise ValueError(f"benchmark case {index} has no profiled samples")
        totals: list[float] = []
        for sample_index, sample in enumerate(case["samples"]):
            if not isinstance(sample, dict) or not isinstance(sample.get("gpu"), dict) or not sample["gpu"]:
                raise ValueError(f"benchmark case {index} sample {sample_index} has no GPU timestamp profile")
            values = list(sample["gpu"].values())
            if any(not finite_number(value) or float(value) < 0 for value in values):
                raise ValueError(f"benchmark case {index} sample {sample_index} contains an invalid GPU timing")
            totals.append(sum(float(value) for value in values))
        median = statistics.median(totals)
        if not math.isfinite(median) or median <= 0:
            raise ValueError(f"benchmark case {index} has an invalid profiled median")
        medians.append(median)
    metric = math.exp(sum(math.log(value) for value in medians) / len(medians))
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("benchmark geometric mean is invalid")
    result["metricMs"] = metric
    result["metric"] = "geometric mean of per-case median summed GPU kernel milliseconds"
    result["profiledCaseMediansMs"] = medians
    return metric


def parity_metric(result: dict, max_dp: float) -> float:
    if result.get("backend") != "webgpu":
        raise ValueError(f"parity backend is {result.get('backend')!r}, expected 'webgpu'")
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


def latency_metric(result: dict) -> float:
    if result.get("backend") != "webgpu":
        raise ValueError(f"latency backend is {result.get('backend')!r}, expected 'webgpu'")
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


def cache_metric(result: dict) -> float:
    if result.get("backend") != "webgpu":
        raise ValueError("cache check did not use WebGPU")
    for key in ("hitVsMiss", "extVsFresh"):
        if not finite_number(result.get(key)) or result[key] != 0:
            raise ValueError(f"cache {key} must be exactly zero, got {result.get(key)!r}")
    stats = result.get("stats", {})
    if any(stats.get(key) != 1 for key in ("hits", "misses", "extensions")):
        raise ValueError(f"cache check did not exercise a hit, miss and extension: {stats}")
    return 0.0


def poll_chrome(cdp: str, url: str, expression: str, kind: str, timeout: float) -> tuple[object, dict]:
    """Loads `url` in a fresh context of a running Chrome and polls `expression` until it is done."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(cdp)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        result: object = None
        try:
            page = context.new_page()
            # Chrome slows covered windows: size this one and bring it to the front
            cdp_session = context.new_cdp_session(page)
            window = cdp_session.send("Browser.getWindowForTarget")
            cdp_session.send("Browser.setWindowBounds", {"windowId": window["windowId"], "bounds": {"width": 1280, "height": 900, "windowState": "normal"}})
            page.bring_to_front()
            page.goto(url, timeout=timeout * 1000)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                result = page.evaluate(f"() => {{ {expression} }}")
                if result_is_done(result, kind):
                    break
                time.sleep(0.25)
        except Exception as error:
            result = {"error": f"browser runner: {error}"}
        finally:
            context.close()
            version = browser.version
            browser.close()  # over CDP this only disconnects
    return result, {"browserName": "chrome", "browserVersion": version}


def main() -> int:
    args = parse_args()
    url = args.url or DEFAULT_URLS[args.result]
    result_expression = f"return window.{RESULT_GLOBALS[args.result]} || null;"
    if args.cdp:
        result, capabilities = poll_chrome(args.cdp, url, result_expression, args.result, args.timeout)
        return finish(args, result, capabilities)
    options = Options()
    options.binary_location = os.environ.get("FIREFOX_BIN", "/usr/bin/firefox")
    options.set_preference("dom.webgpu.enabled", True)
    options.set_preference("dom.webgpu.workers.enabled", True)
    options.add_argument("-no-remote")
    if not args.headed:
        options.add_argument("-headless")
    service = Service(log_output=os.devnull)
    browser = webdriver.Firefox(options=options, service=service)
    browser.set_page_load_timeout(args.timeout)
    browser.set_script_timeout(args.timeout)
    deadline = time.monotonic() + args.timeout
    result: object = None
    capabilities = dict(browser.capabilities)
    try:
        browser.get(url)
        while time.monotonic() < deadline:
            result = browser.execute_script(result_expression)
            if result_is_done(result, args.result):
                break
            time.sleep(0.25)
    except Exception as error:
        result = {"error": f"browser runner: {error}"}
    finally:
        browser.quit()
    return finish(args, result, capabilities)


def finish(args: argparse.Namespace, result: object, capabilities: dict) -> int:
    """Validates and saves a page result; prints the scalar metric last."""
    if not isinstance(result, dict):
        result = {"error": f"{args.result} page did not expose a completed result before timeout"}
    runner = {
        "browser": capabilities.get("browserName"),
        "browserVersion": capabilities.get("browserVersion"),
        "headless": not args.headed and not args.cdp,
        "vkDriverFiles": os.environ.get("VK_DRIVER_FILES"),
    }
    result["runner"] = runner
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        return 1
    try:
        if args.result == "gpu":
            if result.get("status") != "done":
                raise ValueError("GPU benchmark did not finish")
            if result.get("backend") != "webgpu":
                raise ValueError(f"unexpected backend: {result.get('backend')!r}")
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
            metric = bench_metric(result)
        elif args.result == "latency":
            metric = latency_metric(result)
        elif args.result == "parity":
            metric = parity_metric(result, args.max_dp)
        elif args.result == "cache":
            metric = cache_metric(result)
        else:
            metric = kernels_metric(result)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
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
