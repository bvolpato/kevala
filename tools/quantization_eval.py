#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.6,<3"]
# ///
"""Measure bounded weight-only quantization error on cached safetensors tensors.

This is a tensor-level experiment. It does not run a model, evaluate generated
answers, or make a WebGPU performance claim. The default input discovery only
uses complete Hugging Face cache snapshots at the pinned commits below; it
never follows a moving ref or chooses an arbitrary cached snapshot, and it
never downloads weights.

Examples:

  uv run tools/quantization_eval.py
  uv run tools/quantization_eval.py \
      --model qwen-0.8b=/path/to/snapshot \
      --timing-repeats 3 \
      --output docs/benchmarks/quantization-linux.json

The existing Kevala Q8 format is symmetric absmax, one f32 scale per block of
32 values, with ties rounded away from zero. The FP8 comparison uses the CPU
implementation of torch.float8_e4m3fn and the same block shape. MXFP8 uses an
E4M3 payload with one E8M0 power-of-two scale per block of 32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_VERSION = "2026-09-22.2"
BLOCK_SIZE = 32
FP8_E4M3_MAX = 448.0
# These are the exact revisions recorded by the checked artifact. Keep the
# default set pinned so a later cache refresh cannot silently change the
# tensors behind a published comparison. Laya was not present in that
# artifact and remains available through an explicit --model path only.
MODEL_DISCOVERY = (
    (
        "semif-qwen3.5-0.8b",
        "Qwen/Qwen3.5-0.8B",
        "Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
    ),
    (
        "semif-qwen3.5-2b",
        "Qwen/Qwen3.5-2B",
        "Qwen3.5-2B",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
    (
        "semif-qwen3.5-4b",
        "Qwen/Qwen3.5-4B",
        "Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    (
        "kev-9b-base",
        "Qwen/Qwen3.5-9B-Base",
        "Qwen3.5-9B-Base",
        "68c46c4b3498877f3ef123c856ecfde50c39f404",
    ),
    (
        "gemma-4-e2b-it",
        "google/gemma-4-E2B-it",
        "gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    ),
    (
        "gemma-4-e4b-it",
        "google/gemma-4-E4B-it",
        "gemma-4-E4B-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
    ),
)

SAFE_TENSOR_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}
LAYER_PATTERN = re.compile(r"(?:layers?|blocks?|h)\.(\d+)(?:\.|$)")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TensorSource:
    """A tensor entry whose payload can be read from a safetensors row slice."""

    key: str
    path: Path
    snapshot_root: Path | None
    snapshot_revision: str | None
    dtype_name: str
    dtype: torch.dtype
    itemsize: int
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    category: str
    layer: int | None
    stage: str

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def row_values(self) -> int:
        return math.prod(self.shape[1:])

    @property
    def values(self) -> int:
        return math.prod(self.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        metavar="LABEL=PATH",
        help="Safetensors file or HF snapshot directory. May be repeated.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=default_cache_root(),
        help="Hugging Face hub cache root used by automatic discovery.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write compact JSON evidence to this path.",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-tensors", type=int, default=24)
    parser.add_argument("--rows-per-tensor", type=int, default=16)
    parser.add_argument(
        "--max-values-per-model",
        type=int,
        default=4_000_000,
        help="Upper bound for source values sampled from each model.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=0,
        help="CPU conversion timing repeats. Zero omits timings.",
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Stream-hash explicit files whose HF blob name is unavailable.",
    )
    return parser.parse_args()


def default_cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface" / "hub"


def git_revision() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def cache_repo_dir(cache_root: Path, repo_id: str) -> Path:
    return cache_root / ("models--" + repo_id.replace("/", "--"))


def complete_snapshot(cache_root: Path, repo_id: str, revision: str) -> Path | None:
    """Return only the complete cache snapshot at the requested commit.

    Do not consult refs/main or fall back to another snapshot. The caller
    supplies a revision recorded by the published evidence.
    """

    root = cache_repo_dir(cache_root, repo_id)
    candidate = root / "snapshots" / revision
    if not candidate.is_dir():
        return None
    return candidate if valid_safetensor_files(candidate) else None


def discover_models(cache_root: Path) -> tuple[list[tuple[str, str, Path, str]], list[str]]:
    found: list[tuple[str, str, Path, str]] = []
    missing: list[str] = []
    for label, repo_id, _, revision in MODEL_DISCOVERY:
        snapshot = complete_snapshot(cache_root, repo_id, revision)
        if snapshot is None:
            missing.append(f"{label} ({repo_id}@{revision})")
        else:
            found.append((label, repo_id, snapshot, f"automatic HF cache discovery at pinned revision {revision}"))
    return found, missing


def parse_model_arg(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec:
        raise ValueError(f"--model must be LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label:
        raise ValueError(f"--model label is empty in {spec!r}")
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label}: input path does not exist: {path}")
    return label, infer_repo_id(path), path


def infer_repo_id(path: Path) -> str:
    for part in path.resolve().parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return "local"


def valid_safetensor_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".safetensors" and path.stat().st_size > 8 else []
    if not path.is_dir():
        return []
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            expected = sorted({path / str(name) for name in weight_map.values()})
        except (OSError, json.JSONDecodeError, AttributeError):
            return []
        if not expected or any(not file.is_file() or file.stat().st_size <= 8 for file in expected):
            return []
        return expected
    candidates = sorted(path.glob("*.safetensors"))
    return [candidate for candidate in candidates if candidate.is_file() and candidate.stat().st_size > 8]


def snapshot_info(path: Path, cache_root: Path) -> tuple[Path | None, str | None]:
    # HF snapshots contain symlinks into the blob store. Inspect the
    # un-resolved snapshot path first so the commit remains available.
    for candidate in (path, path.resolve()):
        for index, part in enumerate(candidate.parts):
            if part == "snapshots" and index + 1 < len(candidate.parts):
                return Path(*candidate.parts[: index + 2]), candidate.parts[index + 1]
    return None, None


def json_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: safetensors header is truncated")
        header_size = int.from_bytes(prefix, "little")
        if header_size <= 0 or header_size > 128 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header length {header_size}")
        raw = stream.read(header_size)
    if len(raw) != header_size:
        raise ValueError(f"{path}: safetensors header is truncated")
    return json.loads(raw), 8 + header_size


def tensor_category(key: str) -> str:
    lower = key.lower()
    if any(token in lower for token in ("embed_tokens", "embedding", "tok_embeddings", "word_embeddings")):
        return "embedding"
    if "lm_head" in lower or lower.endswith(".output.weight") or lower == "output.weight":
        return "output"
    if any(token in lower for token in ("expert", "experts")):
        return "expert"
    if any(token in lower for token in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "qkv")):
        return "attention"
    if any(token in lower for token in ("mlp", "ffn", "gate_proj", "up_proj", "down_proj", "fc1", "fc2")):
        return "mlp"
    return "other"


def tensor_layer(key: str) -> int | None:
    match = LAYER_PATTERN.search(key)
    return int(match.group(1)) if match else None


def layer_stage(layer: int | None, max_layer: int) -> str:
    if layer is None:
        return "unlayered"
    if max_layer <= 0:
        return "early"
    fraction = layer / max_layer
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def load_tensor_sources(
    path: Path,
    *,
    cache_root: Path,
    max_layer: int,
) -> list[TensorSource]:
    files = valid_safetensor_files(path)
    if not files:
        raise ValueError(f"{path}: no complete .safetensors files found")
    entries: list[tuple[Path, dict[str, Any], int, Path | None, str | None]] = []
    for source_file in files:
        header, data_base = json_header(source_file)
        snapshot_root, revision = snapshot_info(source_file, cache_root)
        entries.append((source_file, header, data_base, snapshot_root, revision))

    sources: list[TensorSource] = []
    for source_file, header, data_base, snapshot_root, revision in entries:
        for key, spec in header.items():
            if key == "__metadata__" or not isinstance(spec, dict):
                continue
            dtype_name = spec.get("dtype")
            shape = tuple(int(value) for value in spec.get("shape", ()))
            offsets = spec.get("data_offsets")
            if dtype_name not in SAFE_TENSOR_DTYPES or len(shape) != 2 or not offsets:
                continue
            if any(value <= 0 for value in shape) or shape[-1] < BLOCK_SIZE:
                continue
            dtype, itemsize = SAFE_TENSOR_DTYPES[dtype_name]
            start, end = (int(offsets[0]), int(offsets[1]))
            expected = math.prod(shape) * itemsize
            if end <= start or end - start != expected:
                continue
            layer = tensor_layer(key)
            sources.append(
                TensorSource(
                    key=key,
                    path=source_file,
                    snapshot_root=snapshot_root,
                    snapshot_revision=revision,
                    dtype_name=dtype_name,
                    dtype=dtype,
                    itemsize=itemsize,
                    shape=shape,
                    data_start=data_base + start,
                    data_end=data_base + end,
                    category=tensor_category(key),
                    layer=layer,
                    stage=layer_stage(layer, max_layer),
                )
            )
    return sources


def choose_sources(
    sources: list[TensorSource],
    *,
    max_tensors: int,
) -> list[TensorSource]:
    if not sources:
        return []
    category_order = {"embedding": 0, "attention": 1, "mlp": 2, "expert": 3, "output": 4, "other": 5}
    stage_order = {"early": 0, "middle": 1, "late": 2, "unlayered": 3}
    groups: dict[tuple[str, str], list[TensorSource]] = {}
    for source in sources:
        groups.setdefault((source.category, source.stage), []).append(source)
    for group in groups.values():
        group.sort(key=lambda source: (source.key, source.shape))

    ordered_groups = sorted(groups.items(), key=lambda item: (
        category_order.get(item[0][0], 99),
        stage_order.get(item[0][1], 99),
    ))
    selected: list[TensorSource] = []
    # One representative per category/stage first gives early, middle, late
    # coverage even when the tensor inventory is dominated by MLP projections.
    for _, group in ordered_groups:
        if len(selected) >= max_tensors:
            break
        selected.append(group[0])
    if len(selected) < max_tensors:
        leftovers = [
            source
            for _, group in ordered_groups
            for source in group[1:]
        ]
        leftovers.sort(key=lambda source: (category_order.get(source.category, 99), source.key))
        selected.extend(leftovers[: max_tensors - len(selected)])
    return selected


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def choose_rows(row_count: int, requested: int, seed: int, key: str) -> list[int]:
    count = min(row_count, max(1, requested))
    if count == row_count:
        return list(range(row_count))
    chosen = {0, row_count // 2, row_count - 1}
    rng = random.Random(stable_seed(seed, key))
    while len(chosen) < count:
        chosen.add(rng.randrange(row_count))
    return sorted(chosen)[:count]


def read_rows(source: TensorSource, rows: list[int]) -> torch.Tensor:
    row_bytes = source.row_values * source.itemsize
    if not rows:
        return torch.empty((0, source.row_values), dtype=torch.float32)
    with source.path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            chunks: list[torch.Tensor] = []
            for row in rows:
                start = source.data_start + row * row_bytes
                end = start + row_bytes
                view = memoryview(mapped)[start:end]
                try:
                    raw = torch.frombuffer(bytearray(view), dtype=source.dtype, count=source.row_values)
                    chunks.append(raw.to(torch.float32).clone())
                finally:
                    view.release()
            return torch.stack(chunks)
        finally:
            mapped.close()


def round_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values >= 0, torch.floor(values + 0.5), torch.ceil(values - 0.5))


def reshape_blocks(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = tuple(values.shape)
    if values.numel() % BLOCK_SIZE:
        raise ValueError(f"sample has {values.numel()} values, not divisible by {BLOCK_SIZE}")
    return values.reshape(-1, BLOCK_SIZE), original_shape


def q8_decode(values: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    blocks, original_shape = reshape_blocks(values)
    amax = blocks.abs().amax(dim=1)
    scale = amax / 127.0
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.clamp(round_away_from_zero(blocks / safe_scale[:, None]), -127, 127)
    quantized = torch.where(scale[:, None] > 0, quantized, torch.zeros_like(quantized))
    decoded = (quantized * scale[:, None]).reshape(original_shape)
    return decoded, {
        "payload_bits": 8.0,
        "scale_bits": 32.0,
        "scale_bytes_per_block": 4.0,
        "clipped_fraction": 0.0,
    }


def fp8_decode(values: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    blocks, original_shape = reshape_blocks(values)
    amax = blocks.abs().amax(dim=1)
    scale = amax / FP8_E4M3_MAX
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = torch.clamp(blocks / safe_scale[:, None], -FP8_E4M3_MAX, FP8_E4M3_MAX)
    encoded = normalized.to(torch.float8_e4m3fn)
    decoded = (encoded.to(torch.float32) * scale[:, None]).reshape(original_shape)
    return decoded, {
        "payload_bits": 8.0,
        "scale_bits": 32.0,
        "scale_bytes_per_block": 4.0,
        "clipped_fraction": float(
            (blocks.abs() > safe_scale[:, None] * FP8_E4M3_MAX * (1.0 + 1e-6)).float().mean()
        ),
    }


def mxfp8_decode(values: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    blocks, original_shape = reshape_blocks(values)
    amax = blocks.abs().amax(dim=1)
    ratio = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    # This is the NVIDIA block-scale "round up" policy expressed as the
    # representable E8M0 exponent. Zero blocks use the reserved zero handling
    # below; nonzero exponents are bounded by E8M0's -127..127 range.
    exponent = torch.ceil(torch.log2(ratio)).clamp(-127, 127)
    scale = torch.where(amax > 0, torch.pow(2.0, exponent), torch.zeros_like(amax))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = torch.clamp(blocks / safe_scale[:, None], -FP8_E4M3_MAX, FP8_E4M3_MAX)
    encoded = normalized.to(torch.float8_e4m3fn)
    decoded = (encoded.to(torch.float32) * scale[:, None]).reshape(original_shape)
    return decoded, {
        "payload_bits": 8.0,
        "scale_bits": 8.0,
        "scale_bytes_per_block": 1.0,
        "scale_policy": "2^clamp(ceil(log2(amax/448)), -127, 127); zero blocks use zero scale",
        "clipped_fraction": float((blocks.abs() > safe_scale[:, None] * FP8_E4M3_MAX).float().mean()),
    }


METHODS = {
    "q8_absmax32": q8_decode,
    "fp8_e4m3fn_absmax32": fp8_decode,
    "mxfp8_e4m3fn_e8m0_32": mxfp8_decode,
}


def error_metrics(reference: torch.Tensor, decoded: torch.Tensor) -> dict[str, float]:
    reference = reference.reshape(-1)
    decoded = decoded.reshape(-1)
    delta = decoded - reference
    mse = float(torch.mean(delta * delta))
    reference_rms = float(torch.sqrt(torch.mean(reference * reference)))
    decoded_rms = float(torch.sqrt(torch.mean(decoded * decoded)))
    denominator = max(reference_rms, 1e-30)
    cosine_denominator = max(
        float(torch.linalg.vector_norm(reference)) * float(torch.linalg.vector_norm(decoded)),
        1e-30,
    )
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "normalized_rmse": math.sqrt(mse) / denominator,
        "mae": float(torch.mean(delta.abs())),
        "max_abs_error": float(delta.abs().max()),
        "cosine": float(torch.dot(reference, decoded) / cosine_denominator),
        "reference_rms": reference_rms,
        "decoded_rms": decoded_rms,
    }


def aggregate_metrics(
    totals: dict[str, float],
    reference: torch.Tensor,
    decoded: torch.Tensor,
) -> None:
    reference = reference.reshape(-1)
    decoded = decoded.reshape(-1)
    delta = decoded - reference
    totals["values"] += float(reference.numel())
    totals["squared_error"] += float(torch.sum(delta * delta))
    totals["squared_reference"] += float(torch.sum(reference * reference))
    totals["squared_decoded"] += float(torch.sum(decoded ** 2))
    totals["dot"] += float(torch.dot(reference, decoded))
    totals["max_abs_error"] = max(totals["max_abs_error"], float(delta.abs().max()))


def finalize_aggregate(totals: dict[str, float]) -> dict[str, float]:
    values = max(totals["values"], 1.0)
    mse = totals["squared_error"] / values
    reference_rms = math.sqrt(totals["squared_reference"] / values)
    decoded_rms = math.sqrt(totals["squared_decoded"] / values)
    cosine_denominator = max(
        math.sqrt(totals["squared_reference"] * totals["squared_decoded"]),
        1e-30,
    )
    return {
        "values": int(totals["values"]),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "normalized_rmse": math.sqrt(mse) / max(reference_rms, 1e-30),
        "max_abs_error": totals["max_abs_error"],
        "cosine": totals["dot"] / cosine_denominator,
        "reference_rms": reference_rms,
        "decoded_rms": decoded_rms,
    }


def empty_totals() -> dict[str, float]:
    return {
        "values": 0.0,
        "squared_error": 0.0,
        "squared_reference": 0.0,
        "squared_decoded": 0.0,
        "dot": 0.0,
        "max_abs_error": 0.0,
    }


def file_provenance(path: Path, cache_root: Path, hash_inputs: bool) -> dict[str, Any]:
    stat = path.stat()
    resolved = path.resolve()
    snapshot_root, revision = snapshot_info(path, cache_root)
    blob_name = resolved.name
    record: dict[str, Any] = {
        "snapshot_revision": revision,
        "snapshot_file": str(path.relative_to(snapshot_root)) if snapshot_root and path.is_relative_to(snapshot_root) else str(path),
        "bytes": stat.st_size,
    }
    if HEX_64.fullmatch(blob_name):
        record["sha256"] = blob_name
        record["sha256_source"] = "HF blob filename"
    elif hash_inputs:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        record["sha256"] = digest
        record["sha256_source"] = "streamed file hash"
    return record


def bytes_per_value(method: str) -> float:
    scale_bytes = 1.0 if method.startswith("mxfp8") else 4.0
    return 1.0 + scale_bytes / BLOCK_SIZE


def cpu_timing(
    samples: list[torch.Tensor],
    method: str,
    repeats: int,
) -> dict[str, Any] | None:
    if repeats <= 0 or not samples:
        return None
    flattened = torch.cat([sample.reshape(-1) for sample in samples])
    usable = flattened.numel() - flattened.numel() % BLOCK_SIZE
    if usable < BLOCK_SIZE:
        return None
    flattened = flattened[: min(usable, 1_000_000)]
    flattened = flattened[: flattened.numel() - flattened.numel() % BLOCK_SIZE].contiguous()
    fn = METHODS[method]
    for _ in range(1):
        fn(flattened)
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(flattened)
        durations.append(time.perf_counter() - start)
    median = sorted(durations)[len(durations) // 2]
    return {
        "values": int(flattened.numel()),
        "repeats": repeats,
        "median_seconds": median,
        "median_us_per_million_values": median * 1_000_000 / flattened.numel() * 1_000_000,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "scope": "sampled quantize plus dequantize only",
    }


def evaluate_model(
    *,
    label: str,
    repo_id: str,
    path: Path,
    cache_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"[{label}] indexing {path}", file=sys.stderr)
    # Layer count is inferred from names before stage assignment. A model can
    # use a different naming convention; those tensors remain unlayered.
    provisional = load_tensor_sources(path, cache_root=cache_root, max_layer=0)
    max_layer = max((source.layer or 0 for source in provisional), default=0)
    sources = load_tensor_sources(path, cache_root=cache_root, max_layer=max_layer)
    selected = choose_sources(sources, max_tensors=args.max_tensors)
    if not selected:
        raise ValueError(f"{label}: no floating point matrix tensors with at least {BLOCK_SIZE} columns")

    sampled: list[dict[str, Any]] = []
    method_totals = {method: empty_totals() for method in METHODS}
    method_metadata: dict[str, dict[str, Any]] = {}
    timing_samples: list[torch.Tensor] = []
    values_used = 0
    skipped: list[dict[str, Any]] = []

    for source in selected:
        if source.row_values % BLOCK_SIZE:
            skipped.append({
                "tensor": source.key,
                "reason": f"last dimension {source.row_values} is not divisible by {BLOCK_SIZE}",
            })
            continue
        remaining = args.max_values_per_model - values_used
        if remaining < BLOCK_SIZE:
            break
        row_limit = min(args.rows_per_tensor, source.rows, remaining // source.row_values)
        if row_limit < 1:
            skipped.append({"tensor": source.key, "reason": "model sample budget exhausted"})
            continue
        rows = choose_rows(source.rows, row_limit, args.seed, f"{label}:{source.key}")
        reference = read_rows(source, rows)
        values_used += reference.numel()
        timing_samples.append(reference)
        tensor_result: dict[str, Any] = {
            "tensor": source.key,
            "shape": list(source.shape),
            "source_dtype": source.dtype_name,
            "category": source.category,
            "layer": source.layer,
            "stage": source.stage,
            "rows": rows,
            "sampled_values": reference.numel(),
            "methods": {},
        }
        for method, fn in METHODS.items():
            decoded, metadata = fn(reference)
            metrics = error_metrics(reference, decoded)
            aggregate_metrics(method_totals[method], reference, decoded)
            method_metadata[method] = metadata
            tensor_result["methods"][method] = metrics
        sampled.append(tensor_result)

    files = sorted({source.path for source in sources})
    file_records = [file_provenance(file, cache_root, args.hash_inputs) for file in files]
    snapshot_revision = next((source.snapshot_revision for source in sources if source.snapshot_revision), None)
    source_path = str(path)
    if path.is_relative_to(cache_root):
        source_path = str(path.relative_to(cache_root))

    timings = {
        method: result
        for method in METHODS
        if (result := cpu_timing(timing_samples, method, args.timing_repeats)) is not None
    }
    model_result: dict[str, Any] = {
        "label": label,
        "repo_id": repo_id,
        "source_path": source_path,
        "snapshot_revision": snapshot_revision,
        "files": file_records,
        "sampling": {
            "seed": args.seed,
            "max_tensors": args.max_tensors,
            "selected_tensors": len(selected),
            "sampled_tensors": len(sampled),
            "rows_per_tensor": args.rows_per_tensor,
            "max_values_per_model": args.max_values_per_model,
            "values": values_used,
            "strategy": "deterministic first/middle/last row plus seeded rows, one representative per category/stage",
        },
        "format": {
            "block_values": BLOCK_SIZE,
            "q8_absmax32": {
                "payload": "int8",
                "scale": "f32",
                "bytes_per_value": bytes_per_value("q8_absmax32"),
            },
            "fp8_e4m3fn_absmax32": {
                "payload": "float8_e4m3fn",
                "scale": "f32",
                "bytes_per_value": bytes_per_value("fp8_e4m3fn_absmax32"),
            },
            "mxfp8_e4m3fn_e8m0_32": {
                "payload": "float8_e4m3fn",
                "scale": "e8m0",
                "bytes_per_value": bytes_per_value("mxfp8_e4m3fn_e8m0_32"),
            },
            "alignment_and_container_overhead_included": False,
        },
        "aggregate": {
            method: finalize_aggregate(totals)
            for method, totals in method_totals.items()
        },
        "method_details": method_metadata,
        "timings": timings,
        "tensors": sampled,
        "skipped": skipped,
    }
    if label == "kev-9b-base":
        model_result["interpretation"] = (
            "This is the Qwen3.5-9B-Base checkpoint used by Kev-9B. "
            "The cached Kev adapter is not a merged full-weight checkpoint and was not sampled."
        )
    return model_result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    requested = args.model or []
    models: list[tuple[str, str, Path, str]] = []
    missing: list[str] = []
    if requested:
        for spec in requested:
            label, repo_id, path = parse_model_arg(spec)
            models.append((label, repo_id, path, "explicit input"))
    else:
        models, missing = discover_models(args.cache_root)
        if missing:
            print(
                "[discovery] skipped unavailable or incomplete snapshots: " + ", ".join(missing),
                file=sys.stderr,
            )
    if not models:
        raise SystemExit("No complete safetensors snapshots found. Pass --model LABEL=PATH after downloads finish.")

    report: dict[str, Any] = {
        "schema": "kevala.quantization-evidence.v1",
        "script_version": SCRIPT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": git_revision(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_num_threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
        },
        "experiment": {
            "purpose": "bounded weight-only tensor error comparison",
            "methods": list(METHODS),
            "block_values": BLOCK_SIZE,
            "source_dtype_policy": sorted(SAFE_TENSOR_DTYPES),
            "no_end_to_end_quality": True,
            "no_gpu_measurement": True,
            "no_weight_download": True,
        },
        "models": [],
    }
    if missing:
        report["discovery_skipped"] = missing
    for label, repo_id, path, origin in models:
        result = evaluate_model(
            label=label,
            repo_id=repo_id,
            path=path,
            cache_root=args.cache_root,
            args=args,
        )
        result["input_origin"] = origin
        report["models"].append(result)
    return report


def main() -> int:
    args = parse_args()
    if args.max_tensors < 1 or args.rows_per_tensor < 1 or args.max_values_per_model < BLOCK_SIZE:
        raise SystemExit("max-tensors, rows-per-tensor, and max-values-per-model must be positive")
    if args.timing_repeats < 0:
        raise SystemExit("timing-repeats cannot be negative")
    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
