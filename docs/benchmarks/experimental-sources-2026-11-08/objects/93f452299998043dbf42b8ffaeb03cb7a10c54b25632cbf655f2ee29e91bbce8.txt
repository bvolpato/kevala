#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["selenium>=4.25,<5", "websockets>=13"]
# [tool.uv]
# required-version = ">=0.13.4"
# exclude-newer = "P7D"
# ///
"""Private Laya result adapter using the frozen Q8 browser and containment helpers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked_module(path, expected, name):
    path = path.resolve(strict=True)
    require(path.stat().st_size <= 1024 * 1024, f"Frozen source exceeded its read bound: {path}")
    data = path.read_bytes()
    require(len(data) <= 1024 * 1024 and hashlib.sha256(data).hexdigest() == expected, f"Frozen source identity differs: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Could not import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shared-wrapper", required=True, type=Path)
    parser.add_argument("--shared-wrapper-sha256", required=True)
    parser.add_argument("--session-bus-helper", required=True, type=Path)
    parser.add_argument("--session-bus-helper-sha256", required=True)
    parser.add_argument("--scope-unit", required=True)
    parser.add_argument("--node-owner-pid", required=True, type=int)
    parser.add_argument("--node-owner-start-ticks", required=True)
    parser.add_argument("--chrome-diagnostics-dir", required=True, type=Path)
    parser.add_argument("--binary")
    parser.add_argument("--timeout", type=int, default=660)
    args = parser.parse_args()
    require(1 <= args.timeout <= 660 and not args.output.exists(), "Timeout must be bounded and output must be new")
    shared = checked_module(args.shared_wrapper, args.shared_wrapper_sha256, "kevala_frozen_browser_helpers")
    bus_helper = checked_module(args.session_bus_helper, args.session_bus_helper_sha256, "kevala_frozen_private_bus")
    args.browser, args.backend, args.headed = "chromium", "webgpu", False
    args.chromium_arg = ["--enable-gpu", "--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface"]
    # The shared factory uses this legacy discriminator only for pageLoadStrategy=none.
    # Laya retains its own terminal predicate and result parser below.
    args.result = "q8-swiglu"
    args.cdp = None
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) > 1, "Normal inherited scheduling is required; a CPU7-only process is ineligible")
    containment = {"sessionBus": {}, "scope": None, "snapshots": [], "appScopesBefore": None, "appScopesAfter": None,
        "newAppScopes": None, "cpuAllowed": affinity, "cpuAffinityChanged": False, "continuousContainmentClaim": False,
        "assurance": "reused frozen scope/ancestry helpers; sampled exact membership at bookends; normal inherited CPU scheduling"}
    result = {"status": "running", "backend": "webgpu"}
    runner = {"browserRequested": "fresh Chromium", "sharedWrapper": str(args.shared_wrapper.resolve()),
        "sharedWrapperSha256": args.shared_wrapper_sha256, "sessionBusHelperSha256": args.session_bus_helper_sha256,
        "containment": containment, "existingBrowserAttached": False, "inheritedCdpSettingIgnored": bool(os.environ.get("KEVALA_BENCH_CDP")),
        "chromiumArgs": list(args.chromium_arg), "vkDriverFiles": os.environ.get("VK_DRIVER_FILES")}
    browser = private_bus = diagnostic_paths = diagnostic_state = None
    full_result_fetched = False
    anchors = {}

    def save():
        args.output.write_text(json.dumps({**result, "runner": runner}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def retain_snapshot_partial(error):
        partial = getattr(error, "q8_partial_snapshot", None)
        if partial is not None:
            stored = (containment.get("scope") or {}).get("snapshotFailures", [])
            if any(item is partial or item == partial for item in stored):
                raw = json.dumps(partial, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode()
                containment.setdefault("snapshotFailureReferences", []).append({"stage": partial.get("stage"),
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "storedUnder": "containment.scope.snapshotFailures"})
            else:
                failures = containment.setdefault("snapshotFailures", [])
                if partial not in failures:
                    failures.append(partial)

    def stopped(signum, _frame):
        raise RuntimeError(f"Owned Laya controller received signal {signum}")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        private_bus = bus_helper.PrivateSessionBus(containment["sessionBus"])
        private_bus.check("before-browser-create")
        # The imported create_browser passes this only to ChromeService(env=...).
        args.chrome_service_env = private_bus.service_environment(dict(os.environ))
        diagnostic_paths = shared._chrome_diagnostic_paths(args.chrome_diagnostics_dir)
        args.chrome_diagnostic_paths = diagnostic_paths
        diagnostic_state = {"directory": str(diagnostic_paths["directory"]), "budgetBytes": shared.CHROME_DIAGNOSTIC_BUDGET_BYTES,
            "sampleIntervalMs": 250, "sampleCount": 0, "lastSampleStage": None, "lastSample": None, "peakBytes": 0,
            "budgetExceeded": None, "retained": True, "level": {"chromeDriver": "WARNING", "chromeGpu": "WARNING"}}
        containment["diagnosticLogs"] = diagnostic_state
        scope = shared._q8_scope(args.scope_unit)
        containment["scope"] = scope
        lineage, pid = [], os.getpid()
        while pid > 1 and len(lineage) < 64:
            row = shared._q8_process(pid)
            lineage.append(pid)
            pid = row["ppid"]
        owner = shared._q8_process(args.node_owner_pid)
        require(args.node_owner_pid in lineage and owner["startTicks"] == args.node_owner_start_ticks and owner["cgroup"] == scope["cgroup"], "The declared Node lease owner is not the live controller ancestor")
        runner["nodeLeaseOwner"] = {"before": owner, "meaning": "ancestor identity only; the Node owner records its own inherited-fd FLOCK proofs"}
        containment["appScopesBefore"] = shared._q8_app_scopes()
        containment["snapshots"].append(shared._q8_scope_snapshot(scope, "before-browser-create", {os.getpid()}))
        browser = shared.create_browser(args)
        shared._q8_check_diagnostic_budget(diagnostic_paths, diagnostic_state, "after-browser-create")
        browser.set_page_load_timeout(args.timeout)
        browser.set_script_timeout(args.timeout)
        runner["browserVersion"] = browser.capabilities.get("browserVersion")
        require(str(runner["browserVersion"]).startswith("156."), "This fixed campaign requires Chrome 156")
        driver_pid = browser.service.process.pid
        browser_pid = int(browser.capabilities["goog:processID"])
        roots = {os.getpid(), driver_pid, browser_pid}
        anchors = {"driver": shared._q8_process(driver_pid), "browser": shared._q8_process(browser_pid)}
        containment["snapshots"].append(shared._q8_scope_snapshot(scope, "after-browser-create", roots))
        private_bus.check("after-browser-create")
        containment["snapshots"].append(shared._q8_scope_snapshot(scope, "before-navigation", roots))
        save()
        deadline = time.monotonic() + args.timeout
        browser.get(args.url)
        terminal = False
        while time.monotonic() < deadline:
            shared._q8_check_diagnostic_budget(diagnostic_paths, diagnostic_state, "terminal-poll")
            # One boolean only during timing. Fetch the complete result exactly once.
            done = browser.execute_script("const x=window.gpuBench; return !!(x && (x.done === true || x.error));")
            if done:
                terminal = True
                break
            time.sleep(0.25)
        full_result_fetched = True
        runner["fullResultFetchesAttempted"] = 1
        result = browser.execute_script("return window.gpuBench || null;")
        page_error = isinstance(result, dict) and (result.get("status") == "error" or result.get("error"))
        runner["terminalOutcome"] = "controller-deadline" if not terminal else "page-error" if page_error else "page-completed"
        shared._q8_check_diagnostic_budget(diagnostic_paths, diagnostic_state, "after-full-result")
        snapshot = shared._q8_scope_snapshot(scope, "after-benchmark-result", roots)
        containment["snapshots"].append(snapshot)
        rows = {row["pid"]: row for row in snapshot["processes"]}
        # Keep scope-wide evidence, but quit anchors belong only to the browser tree.
        browser_tree = shared._q8_descendants(rows, {driver_pid, browser_pid})
        anchors = {f"browser-tree-{pid}": rows[pid] for pid in browser_tree if pid in rows}
        private_bus.check("after-benchmark-result")
        require(terminal, "Laya page did not finish before the controller deadline")
        if page_error:
            raise ValueError("Laya page reported error: " + str(result.get("error") or result.get("code")))
        require(isinstance(result, dict) and result.get("status") == "done" and result.get("done") is True, "Laya terminal result did not report completed status")
        shared.validate_backend(result, "webgpu")
        require(result.get("correctness", {}).get("ok") is True and result.get("protocol", {}).get("status") == "complete", "Laya numerical protocol failed")
        require(len(result["protocol"].get("quartets", [])) == 9 and len(result["protocol"].get("aa", [])) == 9, "Laya request sampling is incomplete")
        require(len(result["protocol"].get("pairedKernel", {}).get("quartets", [])) == 9, "Laya paired-kernel sampling is incomplete")
        require(shared.finite_number(result.get("metricMs")) and result["metricMs"] > 0, "Laya baseline metric is invalid")
    except Exception as error:
        retain_snapshot_partial(error)
        if browser is not None and not full_result_fetched:
            try:
                full_result_fetched = True
                runner["fullResultFetchesAttempted"] = 1
                partial = browser.execute_script("return window.gpuBench || null;")
                if isinstance(partial, dict):
                    result = partial
            except Exception:
                pass
        if not isinstance(result, dict):
            result = {"pagePartial": result}
        result.update({"status": "error", "done": True, "controllerError": str(error)})
    finally:
        if browser is not None:
            try:
                browser.quit()
            except Exception as error:
                result.update({"status": "error", "browserCleanupError": str(error)})
            try:
                containment["snapshots"].append(shared._q8_scope_snapshot(scope, "after-browser-quit", {os.getpid()}, anchors))
                containment["appScopesAfter"] = shared._q8_app_scopes()
                before = {row["unit"] for row in containment["appScopesBefore"]}
                containment["newAppScopes"] = [row for row in containment["appScopesAfter"] if row["unit"] not in before]
                require(not containment["newAppScopes"], "Chrome created a sibling app scope")
            except Exception as error:
                retain_snapshot_partial(error)
                containment["error"] = str(error)
                result["status"] = "error"
        if private_bus is not None:
            try:
                private_bus.close()
            except Exception as error:
                containment["sessionBusError"] = str(error)
                result["status"] = "error"
        if diagnostic_paths is not None:
            try:
                shared._q8_check_diagnostic_budget(diagnostic_paths, diagnostic_state, "after-browser-quit")
                files = diagnostic_state["lastSample"]["files"]
                require(all(files[name]["state"] == "present" for name in ("chromeDriver", "chromeGpu")), "A required Chrome diagnostic log is missing")
                for name in ("chromeDriver", "chromeGpu"):
                    files[name]["sha256"] = hashlib.sha256(diagnostic_paths[name].read_bytes()).hexdigest()
            except Exception as error:
                diagnostic_state["error"] = str(error)
                result["status"] = "error"
        if "nodeLeaseOwner" in runner:
            try:
                owner_after = shared._q8_process(args.node_owner_pid)
                require(owner_after["startTicks"] == args.node_owner_start_ticks and owner_after["cgroup"] == scope["cgroup"], "Node lease-owner identity changed")
                runner["nodeLeaseOwner"]["after"] = owner_after
            except Exception as error:
                runner["nodeLeaseOwner"]["error"] = str(error)
                result["status"] = "error"
        result["done"] = True
        save()
    return 0 if result.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
