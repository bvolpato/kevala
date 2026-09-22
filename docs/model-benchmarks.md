# Optional model validation, September 22, 2026

These measurements validate the converted packs and the current Tetris policy. They do not
establish state-of-the-art decision accuracy. Raw scores, source revisions, pack hashes, game
outcomes, and every selected placement are in [the results JSON](benchmarks/models-linux-2026-09-22.json).

Hardware: RTX 5070 Ti, 16 GB, NVIDIA 595.71.05, Ryzen 9 9950X3D, Ubuntu, Firefox 152.0.3.
Browser inference used hardware WebGPU through Vulkan. Runs were serialized.

## Published packs

All five new packs, their manifests, the model card, and license notices are published at
[Hugging Face revision `45da415`](https://huggingface.co/bvolpato/kevala-packs/tree/45da41504c6c117eca940103e49dd5eb1c3eab4f).
Public metadata confirms every new pack's byte count and SHA-256, and each public range request
returns its valid pack header. Remote manifests and notices match the local files. The existing
Laya and Kev-0.8B pack hashes are unchanged.

Loading `semif-qwen3.5-0.8b` by name from this hosted revision, with browser caching disabled,
also passed all 12 Firefox GPU reference decisions (maximum score difference 0.03083).
The results JSON records the publication checks and hosted run under `publication`.

## Integration with the latest kernels

After integrating `main` at `5867706`, Firefox parity passed again for all six decoder packs:
Kev-0.8B 13/13, Kev-4B 13/13, Kev-9B 5/5, and each SemIf size 12/12. Maximum score differences
were 0.01022, 0.00995, 0.00470, 0.03083, 0.02899, and 0.02047, respectively.

Chrome 149 on the NVIDIA adapter also passed Kev-4B 13/13 and SemIf 0.8B/4B 12/12 each, with
maximum differences 0.00997, 0.03082, and 0.02049. Default headless Chrome exposed no adapter;
these runs used its [documented Linux Vulkan flags](https://developer.chrome.com/blog/supercharge-web-ai-testing).
The adapter exposed subgroups but not `shader-f16`, so these checks exercised the subgroup
recurrence and attention paths, not tiled f16 attention. The results JSON records the flags,
device features, and complete scores under `integrationValidation`.

The integration preserves the 256-thread gate reduction and both lane recurrence variants.
Tiled attention offsets and dispatch counts now use the larger model dimensions. The Tetris
timings below were captured before this kernel integration and have not been rerun.

## Conversion fidelity

| Pack | Original reference | Matching decisions | Maximum absolute score difference |
|---|---|---:|---:|
| Kev-4B | CUDA BF16, LoRA merged in F32 | 13/13 | 0.00986 |
| Kev-9B | CPU F32 | 5/5 | 0.00481 |
| SemIf-style Qwen3.5-0.8B | CUDA F32 | 12/12 | 0.03108 |
| SemIf-style Qwen3.5-2B | CUDA F32 | 12/12 | 0.02901 |
| SemIf-style Qwen3.5-4B | CUDA BF16 | 12/12 | 0.02006 |

SemIf prompt token IDs match the native reference on all 12 cases, including multilingual text.
The three sizes use the same token sequences. The 0.8B native CPU run also matched all decisions,
with maximum score difference 0.03082. The Q8 SemIf gate allows 0.05 and requires matching choices;
its 0.8B score difference exceeds the 0.03 gate used for Laya. These limits measure quantization
fidelity on these fixtures, not confidence calibration.

The original Qwen models use Transformers' native recurrence for reference scores. Its optional
CUDA acceleration packages were absent, so these measurements do not compare Kevala latency to an
optimized SemIf Python deployment. Four-billion-parameter references use BF16 because the full
F32 reference does not fit the 16 GB GPU. The 9B reference ran on the CPU and covers only two
requests, totaling five questions.

Native CPU checks also passed for Kev-4B (three questions, exact prompt IDs, maximum difference
0.0100) and Kev-9B (five questions, exact prompt IDs, maximum difference 0.0046). The 9B native
process peaked at 8,841,068 KiB RSS. Reading directly into aligned storage avoids a second complete
8.96 GB pack copy. These native checks used one CPU thread; they are correctness checks, not a
claim of competitive CPU latency.

A Firefox WASM smoke check also loaded the 2.13 GB SemIf-style 2B pack and matched its checked
decision (maximum difference 0.00604). Its WASM addresses can exceed 2 GiB even though the pack
allocation itself is smaller; the JS bridge now treats returned wasm32 pointers as unsigned.

## Seeded Tetris

The harness used seeds 1, 2, and 3, a limit of 20 pieces per game, and batches of four candidate
states. Every model received the same landing descriptions and the same clean-stack question.
Game paths diverge after different choices, so candidate counts and later inputs also differ.
Times include scoring every candidate for a piece, and exclude downloading/loading the model.

| Policy/model | Pieces placed | Lines cleared | Mean final holes | Median time per piece |
|---|---:|---:|---:|---:|
| Kev-4B | 60 | 11 | 1.67 | 2.006 s |
| SemIf-style 0.8B | 60 | 2 | 7.00 | 1.901 s |
| SemIf-style 2B | 60 | 10 | 1.00 | 2.393 s |
| SemIf-style 4B | 60 | 3 | 12.67 | 2.101 s |
| Drop in the spawn column, no model | 36 | 0 | 28.67 | n/a |

All three model-driven games survived the piece limit for every model. The simple drop baseline
lost all three games. Kev-4B cleared the most lines; SemIf-style 2B ended with the fewest holes.
The larger SemIf model did not improve this policy. Longer runs, more seeds, and task-specific
evaluation are needed before choosing a general default or claiming a quality improvement.

## Failures caught during validation

- The sharded safetensors reader added the eight-byte header length twice, shifting tensor data.
  Source offsets now derive from each parsed header exactly once.
- Raw Qwen tokenizer JSON and Transformers' constructed tokenizer split multilingual combining
  marks differently. Conversion now materializes `AutoTokenizer` before packing.
- Static 0.8B dimensions and 32-bit pack offsets could not represent the larger models.
  GPU kernels now specialize the dimensions, including grouped DeltaNet heads, while JavaScript
  keeps source offsets precise beyond 4 GiB and rebases the WASM coordinator subset.
- Kev-9B exhausted GPU memory during upload until the loader began draining queued transfers
  every 64 MiB. The same 8.96 GB pack then loaded and passed its GPU parity check.
- Fresh cache entries were being considered as prefix parents before their carries existed.
  Only completed entries may supply a prefix carry. Allocation failures now discard dynamic
  buffers so a smaller retry can rebuild them.

## Reproduce

Build with `pnpm build`, convert the pinned sources as described in [Packs](packs.md), then serve
with `pnpm serve --port=18086`. Large checks are opt-in:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result parity --browser firefox --max-dp 0.05 --timeout 600 \
  --url 'http://127.0.0.1:18086/parity-kev.html#backend=webgpu&pack=local&model=semif-qwen3.5-4b&reference=./tests/fixtures/golden-semif-4b.json&cache=0'

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result tetris --browser firefox --timeout 900 \
  --url 'http://127.0.0.1:18086/dev/tetris-eval.html#model=kev-4b&backend=webgpu&pieces=20&seeds=1,2,3&batch=4'
```

Change the model and matching reference together. Use `policy=drop` for the no-model baseline.
Native and browser parity helpers use the committed small JSON fixtures; routine tests need
tokenizers, not the optional model weights.
