#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.63.0"]
# [tool.uv]
# required-version = ">=0.13.4"
# exclude-newer = "P7D"
# ///
"""Measure actual Reaction Recorder decisions with Chromium's synthetic webcam.

Serve the local decoder and companion first. Run only one large model benchmark
at a time. Pass Linux Vulkan flags with repeated --browser-arg options if needed.
This measures throughput and camera lifecycle, not expression accuracy.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18086/")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--browser-arg", action="append", default=[])
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--kernel", choices=["auto", "generic", "wide"], default="auto")
    parser.add_argument("--camera-fps", type=int, default=60)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.frames < 8 or args.frames > 360:
        parser.error("--frames must be between 8 and 360")
    if args.camera_fps < 1 or args.camera_fps > 120:
        parser.error("--camera-fps must be between 1 and 120")
    output = {"status": "running", "source": "Chromium synthetic camera", "kernel": args.kernel, "requestedCameraFps": args.camera_fps}
    try:
        with sync_playwright() as playwright, tempfile.TemporaryDirectory(prefix="kevala-camera-") as profile:
            context = playwright.chromium.launch_persistent_context(
                profile, executable_path=str(args.binary), headless=True,
                permissions=["camera"],
                args=[f"--use-fake-device-for-media-stream=fps={args.camera_fps}", "--use-fake-ui-for-media-stream", *args.browser_arg],
            )
            try:
                page = context.pages[0]
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(args.base_url.rstrip("/") + "/?model=gemma-4-e2b&backend=webgpu#/reaction")
                page.evaluate("""async kernel => {
                  const {Kevala}=await import('./js/src/index.js');
                  const load=Kevala.load.bind(Kevala);
                  Kevala.load=options=>load({...options,gpuKernel:kernel});
                  const {session}=await import('./app/session.js');
                  window.benchSession=session; await session.boot();
                  session.select('gemma-4-e2b', {enableVision:true});
                }""", args.kernel)
                page.locator(".mchip").click()
                page.locator('[data-act="load"]').click()
                page.wait_for_function("() => ['ready','error'].includes(window.benchSession.status)", timeout=180000)
                info = page.evaluate("async () => {const {session}=await import('./app/session.js');if(!session.ready)throw new Error(session.error);return session.info;}")
                if info["backend"] != "webgpu" or "image" not in info["modalities"]:
                    raise RuntimeError("the loaded model lacks WebGPU image support")
                page.locator(".mchip").click()
                page.evaluate("""async () => {
                  const {session}=await import('./app/session.js');
                  const model=session.kevala, decide=model.decide.bind(model);
                  window.cameraBench={model,samples:[],streams:[],inFlight:0,maxInFlight:0};
                  const getUserMedia=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                  navigator.mediaDevices.getUserMedia=async (...args)=>{
                    const stream=await getUserMedia(...args);window.cameraBench.streams.push(stream);return stream;
                  };
                  model.decide=async (state,questions,options)=>{
                    const b=window.cameraBench, start=performance.now();
                    let hash=2166136261;
                    const pixels=options.parts[0].data.data;
                    for(let i=0;i<pixels.length;i+=97)hash=Math.imul(hash^pixels[i],16777619);
                    b.inFlight++;b.maxInFlight=Math.max(b.maxInFlight,b.inFlight);
                    try {
                      const result=await decide(state,questions,options);
                      const completedAt=performance.now();
                      b.samples.push({budget:options.imageTokens,elapsedMs:completedAt-start,startedAt:start,completedAt,
                        hash:hash>>>0,timing:result.timing,answers:result.answers});
                      return result;
                    } finally {b.inFlight--;}
                  };
                }""")
                phases = []
                for index, budget in enumerate([70, 40, 16, 40]):
                    before = page.evaluate("window.cameraBench.samples.length")
                    page.locator('[data-f="resolution"]').select_option(str(budget))
                    if index == 0:
                        page.locator('[data-act="start"]').click()
                    page.wait_for_function(
                        "([before,budget,count])=>window.cameraBench.samples.slice(before).filter(x=>x.budget===budget).length>=count",
                        arg=[before, budget, args.frames], timeout=90000,
                    )
                    phase = page.evaluate("""([before,budget,count])=>({budget,
                      samples:window.cameraBench.samples.slice(before).filter(x=>x.budget===budget).slice(0,count),
                      displayedFps:document.querySelector('[data-f="fps"]').textContent,
                      displayedLatency:document.querySelector('[data-f="latency"]').textContent,
                      resolution:document.querySelector('[data-f="frame-label"]').textContent,
                      videoSize:[document.querySelector('video').videoWidth,document.querySelector('video').videoHeight],
                      trackSettings:document.querySelector('video').srcObject.getVideoTracks()[0].getSettings()})""", [before, budget, args.frames])
                    phases.append(phase)
                page.locator('[data-act="stop"]').click()
                page.wait_for_function("window.cameraBench.inFlight===0", timeout=30000)
                checks = page.evaluate("""async ()=>{
                  const {session}=await import('./app/session.js');
                  return {maxInFlight:window.cameraBench.maxInFlight,
                    sameModel:session.kevala===window.cameraBench.model,
                    videoCleared:document.querySelector('[data-f="video"]').srcObject===null,
                    tracksStopped:window.cameraBench.streams.length>0&&window.cameraBench.streams.every(s=>s.getTracks().every(t=>t.readyState==='ended')),
                    lastTab:document.querySelector('.tabs-nav a:last-child').dataset.route,
                    overflow:document.documentElement.scrollWidth>innerWidth};
                }""")
                cgroup_path = Path("/proc/self/cgroup")
                if cgroup_path.exists():
                    group = next((line.removeprefix("0::") for line in cgroup_path.read_text().splitlines() if line.startswith("0::")), None)
                    if group:
                        directory = Path("/sys/fs/cgroup") / group.lstrip("/")
                        output["resources"] = {name: (directory / name).read_text().strip() for name in
                            ["memory.max", "memory.peak", "memory.events", "memory.swap.max", "cpu.max", "cpu.stat"] if (directory / name).exists()}
                if errors or checks["maxInFlight"] != 1 or not checks["sameModel"] or not checks["videoCleared"] or not checks["tracksStopped"] or checks["lastTab"] != "reaction" or checks["overflow"]:
                    raise RuntimeError(f"camera checks failed: {checks}, page errors: {errors}")
                for phase in phases:
                    warm = phase["samples"][4:]
                    times = sorted(row["elapsedMs"] for row in warm)
                    if len({row["hash"] for row in phase["samples"]}) < 2:
                        raise RuntimeError("synthetic camera frames did not change")
                    phase["summary"] = {"discardedWarmups": 4, "meanMs": sum(times) / len(times),
                        "p50Ms": (times[(len(times)-1)//2] + times[len(times)//2]) / 2,
                        "p95Ms": times[min(len(times)-1, (95 * len(times) + 99)//100 - 1)],
                        "completedFps": (len(warm) - 1) * 1000 / (warm[-1]["completedAt"] - warm[0]["completedAt"])}
                output.update(status="done", browser=context.browser.version, info=info, phases=phases, checks=checks, pageErrors=errors)
                page.evaluate("async () => (await import('./app/session.js')).session.kevala.dispose()")
            finally:
                context.close()
    except Exception as error:
        output.update(status="error", error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "phases": [{"budget": row["budget"], **row["summary"]} for row in output["phases"]]}))


if __name__ == "__main__":
    main()
