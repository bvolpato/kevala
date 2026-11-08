#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["selenium>=4.25,<5", "websockets>=13"]
# [tool.uv]
# required-version = ">=0.13.4"
# exclude-newer = "P7D"
# ///
"""Capture WebGPU feature metadata with the reviewed browser lifecycle helpers."""
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
from urllib.parse import urlsplit

CAMPAIGN = Path("/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn")
SHARED = (CAMPAIGN / "shared-browser-boundary-v4/bench-gpu.py", "eee15de6ed7e26d4329d9bdd4cb0653dcf62cad5b45d44f01cce5e110a599ff0")
BUS = (CAMPAIGN / "campaign/clef-affinity-startup-v1-private-session-bus-r2/session_bus.py", "bac20966653dfaf8cbd7df636c03d02e77ee80b4ac1e9ac9931a6a30bb40c716")
CHROME = "/home/bruno/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
FLAGS = ["--enable-gpu", "--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface"]
TOGGLE = "--enable-dawn-features=vulkan_enable_f16_on_nvidia"
JSON_LIMIT = 16 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked_module(pin, name):
    path, digest = pin
    require(path.resolve(strict=True) == path and path.stat().st_size <= 1024**2, "Frozen helper path or size differs")
    data = path.read_bytes()
    require(len(data) <= 1024**2 and hashlib.sha256(data).hexdigest() == digest, f"Frozen source identity differs: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome-diagnostics-dir", type=Path, required=True)
    parser.add_argument("--scope-unit", required=True)
    parser.add_argument("--node-owner-pid", type=int, required=True)
    parser.add_argument("--node-owner-start-ticks", required=True)
    parser.add_argument("--lane", choices=("baseline", "f16-toggle"), required=True)
    args = parser.parse_args()
    url = urlsplit(args.url)
    require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port and url.path == "/feature-probe" and not url.query and not url.fragment, "Only the owned feature page URL is allowed")
    require(args.output.is_absolute() and not args.output.exists(), "Output must be a new absolute path")
    require(not any(key in os.environ for key in ("VK_DRIVER_FILES", "KEVALA_PROFILE_LOCK_HELD", "KEVALA_BENCH_CDP")), "Forbidden inherited environment setting")
    require(os.environ.get("SE_OFFLINE") == "true", "Selenium Manager must remain offline")
    shared = checked_module(SHARED, "kevala_feature_browser_helpers")
    bus_helper = checked_module(BUS, "kevala_feature_private_bus")
    args.browser, args.backend, args.headed, args.cdp, args.binary = "chromium", "webgpu", False, None, CHROME
    args.result = "q8-swiglu"  # Factory discriminator for pageLoadStrategy=none only.
    args.chromium_arg = FLAGS + ([TOGGLE] if args.lane == "f16-toggle" else [])
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) > 1 and os.getpriority(os.PRIO_PROCESS, 0) == 0 and os.sched_getscheduler(0) == os.SCHED_OTHER, "Normal inherited CPU scheduling is required")
    containment = {"sessionBus": {}, "snapshots": [], "cpuAllowed": affinity, "cpuAffinityChanged": False,
                   "continuousContainmentClaim": False, "assurance": "pinned lifecycle helpers; sampled process and TID identity/cgroup bookends"}
    result = {"status": "running", "scope": "capability-metadata-only", "lane": args.lane,
              "runner": {"containment": containment, "chromiumArgs": args.chromium_arg, "existingBrowserAttached": False,
                         "sharedWrapperSha256": SHARED[1], "sessionBusHelperSha256": BUS[1]}}
    browser = private_bus = diagnostics = diagnostic_state = scope = None
    anchors = {}

    def snapshot(stage, roots, original=None):
        try:
            value = shared._q8_scope_snapshot(scope, stage, roots, original)
            containment["snapshots"].append(value)
            return value
        except Exception as error:
            partial = getattr(error, "q8_partial_snapshot", None)
            if partial is not None:
                containment.setdefault("snapshotFailures", []).append(partial)
            raise

    def save():
        data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        require(len(data) <= JSON_LIMIT, "Feature result JSON exceeded its 16 MiB bound")
        with args.output.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def stopped(signum, _frame):
        raise RuntimeError(f"Feature controller received signal {signum}")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        scope = shared._q8_scope(args.scope_unit)
        containment["scope"] = scope
        lineage, pid = [], os.getpid()
        while pid > 1 and len(lineage) < 64:
            lineage.append(pid)
            pid = shared._q8_process(pid)["ppid"]
        owner = shared._q8_process(args.node_owner_pid)
        require(args.node_owner_pid in lineage and owner["startTicks"] == args.node_owner_start_ticks and owner["cgroup"] == scope["cgroup"], "Declared Node lease owner is not the live scope ancestor")
        result["runner"]["nodeLeaseOwnerBefore"] = owner
        private_bus = bus_helper.PrivateSessionBus(containment["sessionBus"])
        private_bus.check("before-browser-create")
        args.chrome_service_env = private_bus.service_environment(dict(os.environ))
        diagnostics = shared._chrome_diagnostic_paths(args.chrome_diagnostics_dir)
        args.chrome_diagnostic_paths = diagnostics
        diagnostic_state = {"budgetBytes": shared.CHROME_DIAGNOSTIC_BUDGET_BYTES, "sampleCount": 0, "peakBytes": 0, "budgetExceeded": None}
        containment["diagnosticLogs"] = diagnostic_state
        containment["appScopesBefore"] = shared._q8_app_scopes()
        snapshot("before-browser-create", {os.getpid()})
        browser = shared.create_browser(args)
        shared._q8_check_diagnostic_budget(diagnostics, diagnostic_state, "after-browser-create")
        browser.set_page_load_timeout(60)
        browser.set_script_timeout(10)
        version = browser.capabilities.get("browserVersion")
        require(version == "156.0.8078.4", f"Expected Chrome 156.0.8078.4, got {version}")
        driver_pid, browser_pid = browser.service.process.pid, int(browser.capabilities["goog:processID"])
        roots = {os.getpid(), driver_pid, browser_pid}
        anchors = {"driver": shared._q8_process(driver_pid), "browser": shared._q8_process(browser_pid)}
        result["runner"].update({"browserVersion": version, "driverPid": driver_pid, "browserPid": browser_pid})
        snapshot("after-browser-create", roots)
        private_bus.check("after-browser-create")
        snapshot("before-navigation", roots)
        browser.get(args.url)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            shared._q8_check_diagnostic_budget(diagnostics, diagnostic_state, "feature-poll")
            if browser.execute_script("return window.f16FeatureProbe?.done === true;") is True:
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("Feature page did not finish within 60 seconds")
        raw = browser.execute_script("return JSON.stringify(window.f16FeatureProbe);")
        require(isinstance(raw, str) and len(raw.encode()) <= 32768, "Page feature metadata exceeded its 32 KiB bound")
        page = json.loads(raw)
        result["page"] = page
        current = snapshot("after-feature-result", roots)
        rows = {row["pid"]: row for row in current["processes"]}
        tree = shared._q8_descendants(rows, {driver_pid, browser_pid})
        anchors = {f"browser-tree-{pid}": rows[pid] for pid in tree if pid in rows}
        private_bus.check("after-feature-result")
        require(page.get("status") == "complete" and page.get("done") is True, "Feature page error: " + str(page.get("error")))
        require(page.get("scope") == "capability-metadata-only" and page.get("identity") == {"nvidia": True, "blackwell": True, "nonfallback": True}, "Unexpected feature page identity")
        require(page.get("device", {}).get("created") is True and page["device"].get("destroyed") is True, "Feature device lifecycle is incomplete")
        result["status"] = "complete"
    except Exception as error:
        result.update({"status": "error", "error": str(error)[:4096]})
    finally:
        if browser is not None:
            try:
                browser.quit()
                snapshot("after-browser-quit", {os.getpid()}, anchors)
            except Exception as error:
                result.update({"status": "error", "browserCleanupError": str(error)[:4096]})
        if "appScopesBefore" in containment:
            try:
                containment["appScopesAfter"] = shared._q8_app_scopes()
                previous = {row["unit"] for row in containment["appScopesBefore"]}
                containment["newAppScopes"] = [row for row in containment["appScopesAfter"] if row["unit"] not in previous]
                require(not containment["newAppScopes"], "Chrome created a sibling app scope")
            except Exception as error:
                result.update({"status": "error", "appScopeError": str(error)[:4096]})
        if private_bus is not None:
            try:
                private_bus.close()
            except Exception as error:
                result.update({"status": "error", "sessionBusCleanupError": str(error)[:4096]})
        if diagnostics is not None:
            try:
                shared._q8_check_diagnostic_budget(diagnostics, diagnostic_state, "after-browser-quit")
            except Exception as error:
                result.update({"status": "error", "diagnosticError": str(error)[:4096]})
        if scope is not None and "nodeLeaseOwnerBefore" in result["runner"]:
            try:
                owner = shared._q8_process(args.node_owner_pid)
                require(owner["startTicks"] == args.node_owner_start_ticks and owner["cgroup"] == scope["cgroup"], "Node lease-owner identity changed")
                result["runner"]["nodeLeaseOwnerAfter"] = owner
            except Exception as error:
                result.update({"status": "error", "nodeOwnerError": str(error)[:4096]})
        result["done"] = True
        save()
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
