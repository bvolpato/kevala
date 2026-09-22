#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright>=1.50"]
# ///
"""Records the model playing Tetris on the clip stage and renders the social video and README GIF.

Needs the dev server (`node scripts/serve.mjs . --port=8123`), the dev packs in tmp/, ffmpeg, and
the shared agent Chrome on port 9333 (the script opens its own context there and closes it).

    uv run dev/record-tetris.py                 # 50 s capture, then tmp/tetris.mp4 and tmp/tetris.gif
    uv run dev/record-tetris.py --mp4 0:29.2 --gif 0:15.6

The capture is Chrome's screencast at 1920x1080 (a 1280x720 page at 1.5x). The MP4 (H.264, 30 fps)
suits X and other social posts; the GIF (880 px, 15 fps) is docs/tetris.gif.
"""

import argparse
import asyncio
import base64
import json
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8123/?pack=local&clip=1&speed=normal#/tetris"
FRAMES = Path("tmp/clip")
FFMPEG = "ffmpeg"


async def record(seconds: float) -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("f*.jpg"):
        old.unlink()
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp("http://127.0.0.1:9333")
        context = await browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        try:
            page = await context.new_page()
            await page.goto(URL)
            await page.wait_for_function("window.kevala_site && window.kevala_site.session")
            await page.evaluate("window.kevala_site.session.load('laya')")
            await page.wait_for_function("window.kevala_site.session.ready", timeout=120_000)
            await page.wait_for_function("window.tetris && window.tetris.ai.enabled", timeout=60_000)
            # a few decisions first, so the recorded ones run warm; then a fresh game on camera
            await page.wait_for_timeout(6000)
            await page.evaluate("window.tetris.ignoreKeys(); window.tetris.start(true)")

            cdp = await context.new_cdp_session(page)
            frames: list[dict] = []

            async def on_frame(event: dict) -> None:
                name = f"f{len(frames) + 1:05d}.jpg"
                (FRAMES / name).write_bytes(base64.b64decode(event["data"]))
                frames.append({"file": name, "t": event["metadata"]["timestamp"]})
                await cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

            cdp.on("Page.screencastFrame", lambda e: asyncio.ensure_future(on_frame(e)))
            await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 1920, "maxHeight": 1080})
            states = []
            loop = asyncio.get_running_loop()
            start = loop.time()
            while loop.time() - start < seconds:
                states.append(await page.evaluate("({ lines: tetris.game.lines, t: Date.now() / 1000 })"))
                await page.wait_for_timeout(250)
            await cdp.send("Page.stopScreencast")
            await page.wait_for_timeout(300)
            (FRAMES / "frames.json").write_text(json.dumps({"frames": frames, "states": states}))
        finally:
            await context.close()
            await browser.close()  # over CDP this only disconnects

    t0 = frames[0]["t"]
    cleared, prev = [], 0
    for state in states:
        if state["lines"] != prev:
            cleared.append(f"{state['t'] - t0:.1f}s +{state['lines'] - prev}")
            prev = state["lines"]
    print(f"{len(frames)} frames over {frames[-1]['t'] - t0:.1f} s; line clears at {', '.join(cleared) or 'none'}")


def concat_list(segment: str) -> Path:
    """An ffmpeg concat list holding the frames of `start:end` (seconds) at their real durations."""
    start, end = (float(x) for x in segment.split(":"))
    frames = json.loads((FRAMES / "frames.json").read_text())["frames"]
    t0 = frames[0]["t"]
    chosen = [f for f in frames if start <= f["t"] - t0 <= end]
    lines = []
    for a, b in zip(chosen, chosen[1:]):
        lines += [f"file '{(FRAMES / a['file']).resolve()}'", f"duration {b['t'] - a['t']:.4f}"]
    lines.append(f"file '{(FRAMES / chosen[-1]['file']).resolve()}'")
    path = FRAMES / f"list-{start}-{end}.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def render(mp4: str, gif: str) -> None:
    video = ["-vf", "scale=1920:1080:flags=lanczos,fps=30", "-c:v", "libx264", "-preset", "slow", "-crf", "18"]
    video += ["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"]
    ffmpeg = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i"]
    subprocess.run([*ffmpeg, str(concat_list(mp4)), *video, "tmp/tetris.mp4"], check=True)
    palette = (
        "fps=15,scale=880:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run([*ffmpeg, str(concat_list(gif)), "-vf", palette, "-loop", "0", "tmp/tetris.gif"], check=True)
    for out in ("tmp/tetris.mp4", "tmp/tetris.gif"):
        print(f"{out}: {Path(out).stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=50, help="length of the capture")
    parser.add_argument("--mp4", default="0:29.2", help="segment of the capture for the video, start:end in seconds")
    parser.add_argument("--gif", default="0:15.6", help="segment of the capture for the GIF")
    parser.add_argument("--render-only", action="store_true", help="reuse the last capture in tmp/clip")
    args = parser.parse_args()
    if not args.render_only:
        asyncio.run(record(args.seconds))
    render(args.mp4, args.gif)


if __name__ == "__main__":
    main()
