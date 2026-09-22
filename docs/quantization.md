# Quantization choice

Kevala currently uses weight-only Q8 matrices: signed int8 payloads, symmetric
absmax scaling, and one f32 scale for every 32 values in a row. This is the
format implemented by the browser and WASM paths, and it is also the format
used by the checked model packs. The format costs 1.125 bytes per weight before
container or alignment overhead. Q8 is storage-only in Kevala: the kernels
dequantize the int8 payload into floating-point tiles, multiply by floating-point
activations, and accumulate in f32. The packed weights do not imply integer
activation arithmetic or an integer GEMM.

## What was measured

[quantization_eval.py](../tools/quantization_eval.py) compares that format with
two CPU reference encodings:

* fp8_e4m3fn_absmax32: an E4M3FN payload with the same f32 absmax scale per
  32 values.
* mxfp8_e4m3fn_e8m0_32: an E4M3FN payload with one power-of-two E8M0 scale
  per 32 values. Its payload and scale cost 1.03125 bytes per weight.

The Q8 path uses round-to-nearest with ties away from zero, matching the Rust
converter. The FP8 paths use PyTorch's CPU torch.float8_e4m3fn conversion.
All methods dequantize to f32 before the error metrics are computed.

The MXFP8 comparison is one explicit E4M3/E8M0 policy, not a claim about every
MX implementation. For each nonzero block it computes
s = 2^clamp(ceil(log2(amax / 448)), -127, 127), encodes
clamp(x / s, -448, 448) as E4M3FN, and decodes as encoded * s. Zero blocks
use a zero scale. This follows NVIDIA cuDNN's block-scale
[round-up rule](https://docs.nvidia.com/deeplearning/cudnn/v1.10.0/operations/BlockScaling.html)
with E4M3's 448 maximum and E8M0's power-of-two scale. The
[OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
defines the shared scale, element type, and block-size family and allows other
conversion algorithms.

The run is deliberately bounded. It uses at most 24 matrix tensors and 4
million values per model, samples up to 16 rows from each selected tensor, and
chooses the first, middle, last, plus seeded rows. Selection covers embeddings,
attention, MLP, output, expert, and other matrices across early, middle, and
late layers when those names are present. It reads selected rows directly from
the safetensors files with memory mapping, so it does not load a model.

This is tensor reconstruction evidence. It does not measure perplexity,
decision accuracy, model outputs, WebGPU kernels, or end-to-end latency.

Reproduce the checked run after the corresponding pinned HF snapshots are
present:

    uv run tools/quantization_eval.py \
      --max-tensors 24 \
      --rows-per-tensor 16 \
      --max-values-per-model 4000000 \
      --timing-repeats 3 \
      --output docs/benchmarks/quantization-linux-2026-09-22.json

The default cache scan accepts only the six revisions represented in the
checked artifact: Qwen3.5 0.8B at
`2fc06364715b967f1860aea9cf38778875588b17`, Qwen3.5 2B at
`15852e8c16360a2fea060d615a32b45270f8a8fc`, Qwen3.5 4B at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Qwen3.5 9B Base at
`68c46c4b3498877f3ef123c856ecfde50c39f404`, Gemma 4 E2B IT at
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, and Gemma 4 E4B IT at
`ee0ef6023621cff504d758262d4e04895a5af4a2`. It never follows `refs/main`
or selects another cached snapshot. Automatic discovery never downloads
weights. The 9B entry is the pinned Qwen3.5-9B-Base checkpoint used by Kev-9B;
the cached Kev adapter is not a merged full-weight checkpoint and is not
represented as if it were. Gemma 4 E2B and E4B regular IT checkpoints were
sampled; see the [Gemma 4 integration notes](gemma4.md) for their architecture
and validation contract. Laya was not present as a complete safetensors
snapshot in the checked artifact and remains an explicit input when its
original checkpoint is available.

Explicit `--model LABEL=PATH` inputs remain supported for local files or other
snapshots. They bypass the six-model default pin list, and the output records
the discovered snapshot revision, source path, and file identity when
available.

The recorded measurements used evaluator `2026-09-22.1`, whose SHA-256 remains
in the artifact. Evaluator `2026-09-22.2` subsequently pinned default checkpoint
discovery; its quantization arithmetic and sampling are unchanged. The old
measurements have not been relabeled as a run of the updated script.

## Results

The measured aggregate values are weighted over the sampled tensors. Lower
normalized RMSE is better; cosine is closer to 1. Values are rounded here,
while the JSON retains the full precision.

| sampled checkpoint | Q8 NRMSE | FP8 E4M3FN NRMSE | MXFP8 NRMSE | Q8 cosine | FP8 cosine | MXFP8 cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SemIf Qwen3.5 0.8B | 0.008151 | 0.020190 | 0.026352 | 0.99996681 | 0.99979614 | 0.99965274 |
| SemIf Qwen3.5 2B | 0.008339 | 0.019661 | 0.026312 | 0.99996518 | 0.99980669 | 0.99965375 |
| SemIf Qwen3.5 4B | 0.007945 | 0.020277 | 0.026746 | 0.99996840 | 0.99979438 | 0.99964230 |
| Kev-9B base checkpoint | 0.006541 | 0.022578 | 0.026387 | 0.99997860 | 0.99974507 | 0.99965191 |
| Gemma 4 E2B IT | 0.005759 | 0.021846 | 0.026692 | 0.99998350 | 0.99976144 | 0.99964383 |
| Gemma 4 E4B IT | 0.005797 | 0.022092 | 0.026986 | 0.99998331 | 0.99975590 | 0.99963585 |

Using the full-precision values in the JSON, FP8 E4M3FN has 2.36 to 3.81 times
the Q8 normalized RMSE, and MXFP8 has 3.16 to 4.66 times the Q8 normalized
RMSE. All three encodings retain a high cosine on
these individual tensors, but tensor cosine alone cannot establish model
quality.

The optional CPU timings in the JSON are only sampled encode plus decode
conversions. They are not GEMM timings and should not be used as a browser
runtime comparison. The run used PyTorch 2.14.0+cu130 with the CPU device and
16 Torch threads. CUDA being available on the host does not turn this
experiment into a GPU measurement.

## Decision

Keep Q8 as Kevala's default format pending end-to-end validation. This bounded
tensor experiment found lower reconstruction error for Q8 and Kevala already
has a portable implementation on CPU/WASM and WebGPU. It does not establish
model quality by itself. Existing end-to-end conversion fidelity and browser
parity are reported separately in
[model-benchmarks.md](model-benchmarks.md#conversion-fidelity) and the
[architecture fidelity notes](architecture.md#fidelity).
The [decision benchmark](../BENCHMARK.md) separately compares Gemma Q8 with
original BF16 and FP32 outputs. It includes a material E2B probability outlier,
so matching most decisions should not be read as lossless quantization.

FP8 is useful when the target hardware has a native FP8 datapath and the
weights are produced with an appropriate calibration or quantization-aware
training recipe. It is not automatically a better replacement for Q8:

* E4M3 has only three explicit mantissa bits and logarithmically spaced values,
  while Q8 uses a uniform 127-step grid for a fixed absmax scale. The
  logarithmic representation can preserve values closer to zero when a block
  contains an outlier, while the uniform grid can be more accurate over the
  rest of that block. The aggregate direction depends on the value
  distribution; the measured samples favored Q8.
* With an f32 scale, FP8 has the same 1.125 bytes-per-weight storage cost as
  Q8. MXFP8 saves scale bytes, but its power-of-two scale adds another
  approximation, as the measurements show.
* Current WebGPU feature names expose shader-f16 and subgroups, but no
  portable FP8 feature. The [WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/)
  defines f32 and optional f16 numeric types; the [WebGPU feature-name
  registry](https://gpuweb.github.io/types/types/GPUFeatureName.html) does not
  define an FP8 feature. An implementation would therefore need an emulated
  decode path or a vendor-specific extension.
* WGSL's optional
  [packed 4x8 integer dot product extension](https://gpuweb.github.io/gpuweb/wgsl/#packed-4x8-integer-dot-product)
  is an int8 dot-product facility. Kevala's current Q8 kernels instead
  dequantize weights and use floating-point activations and accumulation, so
  that extension does not describe the current path and does not provide an
  FP8 type.

An FP8 pack can be reconsidered as an opt-in format after a native target is
identified and a follow-up experiment measures full model outputs, decision
accuracy, and end-to-end CPU/WebGPU latency. That experiment should compare
per-layer or calibrated scales and mixed precision, rather than treating this
absmax sample as a universal FP8 verdict.

## Primary references

* NVIDIA's [quantized types and scaling schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
  describes FP8 E4M3/E5M2 and block-scaled formats.
* The original [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
  paper explains why E4M3 and E5M2 make different range and precision
  tradeoffs.
* NVIDIA's [accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
  documents the need for calibration and the fact that quantization error is
  model and layer dependent.
* [SmoothQuant](https://arxiv.org/abs/2211.10438) is a primary example of
  changing activation and weight ranges before low-bit quantization; it is
  relevant context for a future W8A8 or FP8 experiment, not part of this
  weight-only measurement.
