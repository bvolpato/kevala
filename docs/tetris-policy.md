# Tetris demo policy and measurements

The auto player searches turn, slide, and drop paths with the game's collision rules. It measures
the resulting boards, favors cleared rows, and penalizes new and existing holes, height, and
roughness. It keeps up to four distinct descriptions close to the best measured outcome. The
selected decision model scores those descriptions and chooses among them. When the shortlist has
one state, the move is determined by the game rules. This keeps clearly worse landings out of the model
batch and avoids presenting many identical descriptions with different board quality.

The controller checks that a chosen landing is still reachable after inference and replans if
gravity blocks a key on the way. It starts scoring the next piece while the current one moves,
and advances an auto drop a row at a time. The panel
reports the **full model wait for one piece**, rather than dividing that wait by the number of
landings searched.

## Seeded WebGPU evaluation, September 23, 2026

These runs used the same local Q8 packs on Chrome 149, Ubuntu, and an NVIDIA RTX 5070 Ti through
Vulkan WebGPU. Baseline was `main` at `16b45947`; the new policy is this change. Each game used
the same seed and piece limit in both runs. The evaluation harness plays without gravity so it
measures model decisions and board quality independently of animation. Its median time is the
full `decide` or `decideMany` wait per piece, excluding model loading. Game paths can diverge
after the first different choice. Runs were serialized on the GPU.

| Model | Seeds × pieces | Policy | Pieces placed | Lines cleared | Mean final holes | Median model wait |
|---|---:|---|---:|---:|---:|---:|
| Laya | 10 × 100 | baseline | 1,000 | 351 | 3.5 | 62.6 ms |
| Laya | 10 × 100 | shortlist | 1,000 | 376 | 1.0 | 8.2 ms |
| Kev 0.8B | 3 × 60 | baseline | 180 | 59 | 2.7 | 88.6 ms |
| Kev 0.8B | 3 × 60 | shortlist | 180 | 62 | 2.0 | 10.5 ms |
| SemIf 2B | 3 × 60 | baseline | 180 | 55 | 2.0 | 433.5 ms |
| SemIf 2B | 3 × 60 | shortlist | 180 | 65 | 1.0 | 33.8 ms |

All games reached their piece limits. [Per-game outcomes, model revisions, runner details, and
aggregate and median decision times](benchmarks/tetris-policy-chrome-2026-09-23.json) are saved
with the measurements.

The sample supports a faster and safer demo on these three models; it is not a general claim about
Tetris skill or other hardware. The placement search covers paths this auto controller drives,
not every reachable path under arbitrary manual play.

To reproduce, serve the baseline and candidate worktrees separately with their local packs in
`tmp/`, then run `scripts/bench-gpu.py` against `dev/tetris-eval.html` on a browser exposing a
hardware WebGPU adapter. For example, with a GPU-enabled Chrome CDP endpoint on port 9333 and
the candidate site on port 18130:

```sh
uv run scripts/bench-gpu.py --result tetris --cdp http://127.0.0.1:9333 --timeout 900 \
  --url 'http://127.0.0.1:18130/dev/tetris-eval.html#model=laya&backend=webgpu&pieces=100&seeds=1,2,3,4,5,6,7,8,9,10&batch=32' \
  --output tmp/tetris-eval.json
```
