#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.6",
#     "transformers>=5.17",
#     "accelerate>=1.12",
#     "huggingface_hub",
#     "safetensors",
#     "numpy",
#     "jinja2",
# ]
# ///
"""Compare Gemma 4 base and IT weights on the same direct-decision rows.

This is a reference protocol and weights ablation.  It is intentionally kept
separate from Kevala's WebGPU timing and quantized pack measurements:

* ``*-base`` uses the regular Gemma checkpoint and an explicit raw-completion
  prompt.  The prompt exposes ``Evidence:``, ``Criterion:``, lettered
  ``Options:`` with Kev typed-choice descriptions, and ends at an ``Answer:``
  prefix.
* ``*-it`` uses Gemma's native chat template with
  ``enable_thinking=False`` and the direct-options instruction.
* Both paths use the same semantic rows, BF16 weights, CUDA device, native
  last-position logits, and no quantization.

The model loader, text-only model construction, persistent-buffer handling,
and CPU-resident PLE path are imported from ``golden_gemma.py`` beside this
script.  Pass ``--gemma-helper`` when running from a checkout where that
helper has not landed yet or lives elsewhere.

The frozen Kevala holdout contains 36 semantic decisions.  The default
``identity``, ``rotate1``, and ``rotate2`` option orders therefore make 108
scored rows.  The expected expanded count can be disabled with
``--expected-rows 0`` for a smaller protocol smoke run; the manifest records
the exact source rows and
digest in either case.

Examples (the commands below perform model loading and CUDA work):

    HF_HOME=tmp/hf uv run tools/benchmark_gemma_variants.py \
      --device cuda --dtype bfloat16 --output tmp/gemma-base-vs-it.json

    uv run tools/benchmark_gemma_variants.py --plan

Local snapshots are accepted for offline runs.  They must include a 40
character snapshot SHA in the path or via the corresponding ``--*-revision``
argument so output metadata never calls an unpinned local checkout pinned.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT_VERSION = "2026-09-22.1"
LETTERS = "ABCDEFGHIJKLMNOP"
SEMIF_METHOD_REVISION = "1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
HEX_SHA = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_DATASET = ROOT / "benchmarks" / "decisions" / "kevala-authored36.jsonl"
DEFAULT_GEMMA_HELPER = HERE / "golden_gemma.py"
DEFAULT_PERMUTATIONS = ("identity", "rotate1", "rotate2")
BASE_PROTOCOL = "raw-completion-v2"
IT_PROTOCOL = "gemma-it-chat-v1"
OPTION_RENDERING = {
    "name": "kev-typed-choice-v1",
    "choice_description": "${id}: ${text}",
    "empty_description": "${id}",
    "source": "Kevala Rust kev::parse_questions and golden_semif._question_options",
    "applies_to": [BASE_PROTOCOL, IT_PROTOCOL],
}

# These are the ``sha`` values returned by the Hugging Face model metadata API
# on 2026-09-22.  The benchmark refuses a remote revision that is not a full
# commit SHA, so a moving ``main``/``latest`` ref cannot enter the evidence.
MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "e2b-base": {
        "family": "E2B",
        "weight_variant": "base",
        "repo": "google/gemma-4-E2B",
        "revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
    },
    "e2b-it": {
        "family": "E2B",
        "weight_variant": "it",
        "repo": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    },
    "e4b-base": {
        "family": "E4B",
        "weight_variant": "base",
        "repo": "google/gemma-4-E4B",
        "revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
    },
    "e4b-it": {
        "family": "E4B",
        "weight_variant": "it",
        "repo": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _typed_choice_description(option_id: str, description: str) -> str:
    """Render an option exactly as Kev's typed choice parser does."""
    if not description:
        return option_id
    return f"{option_id}: {description}"


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_gemma_helper(path: Path) -> ModuleType:
    """Load golden_gemma without copying its loader into this script."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Gemma helper not found at {path}; pass --gemma-helper to the merged golden_gemma.py"
        )
    helper_dir = str(path.parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    spec = importlib.util.spec_from_file_location("kevala_golden_gemma", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Gemma helper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the frozen 36-row Kevala holdout without changing its text."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"canonical holdout not found at {path}; stage benchmarks/decisions/kevala-authored36.jsonl"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(source, dict):
            raise TypeError(f"{path}:{line_number}: decision row must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path}:{line_number}: decision row needs a string id")
        if source_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate decision id {source_id!r}")
        seen.add(source_id)
        state = source.get("state")
        question = source.get("question")
        options = source.get("options")
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError(f"{path}:{line_number} {source_id}: state must be nonempty text, object, or array")
        if not isinstance(question, str) or not question:
            raise ValueError(f"{path}:{line_number} {source_id}: question must be nonempty text")
        if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
            raise ValueError(f"{path}:{line_number} {source_id}: options must contain 2-16 entries")
        copied_options = []
        option_ids: set[str] = set()
        for option in options:
            if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not option["id"]:
                raise ValueError(f"{path}:{line_number} {source_id}: every option needs a nonempty string id")
            if option["id"] in option_ids:
                raise ValueError(f"{path}:{line_number} {source_id}: duplicate option id {option['id']!r}")
            description = option.get("description")
            if not isinstance(description, str) or not description:
                raise ValueError(f"{path}:{line_number} {source_id}: option descriptions must be nonempty text")
            option_ids.add(option["id"])
            copied_options.append(
                {
                    "id": option["id"],
                    "description": _typed_choice_description(option["id"], description),
                }
            )
        gold_id = source.get("gold_id")
        if not isinstance(gold_id, str) or gold_id not in option_ids:
            label = source.get("label")
            if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(copied_options):
                raise ValueError(f"{path}:{line_number} {source_id}: no valid gold option")
            gold_id = copied_options[label]["id"]
        rows.append(
            {
                "id": source_id,
                # The browser benchmark keys expanded rows from the canonical
                # source id. Keep the authored group id available for grouped
                # bootstrap summaries without using it as the row namespace.
                "semantic_id": source_id,
                "group_id": source.get("group_id") or source_id,
                "case_id": source.get("group_id") or source_id,
                "question_id": source_id,
                "family": source.get("family"),
                "state": state,
                "question": question,
                "options": copied_options,
                "gold_id": gold_id,
                "source_label": source.get("label"),
                "provenance": source.get("provenance"),
            }
        )
    if len(rows) != 36:
        raise ValueError(f"{path}: expected 36 canonical holdout rows, found {len(rows)}")
    source_info: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None,
        "sha256": _file_sha256(path),
        "rows": len(rows),
        "semantic_decisions": len({row["semantic_id"] for row in rows}),
    }
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest_path}: invalid JSON: {error}") from error
        kevala_manifest = manifest.get("kevala") if isinstance(manifest, dict) else None
        if isinstance(kevala_manifest, dict):
            expected_sha = kevala_manifest.get("sha256")
            expected_rows = kevala_manifest.get("rows")
            if expected_sha and expected_sha != source_info["sha256"]:
                raise ValueError(f"{path}: SHA-256 does not match manifest: {expected_sha} != {source_info['sha256']}")
            if expected_rows and expected_rows != len(rows):
                raise ValueError(f"{path}: row count does not match manifest: {expected_rows} != {len(rows)}")
            source_info["manifest_path"] = str(manifest_path)
            source_info["manifest_sha256"] = _file_sha256(manifest_path)
            source_info["manifest_version"] = manifest.get("version")
            source_info["manifest_permutations"] = manifest.get("permutations")
            source_info["semif_revision"] = manifest.get("semif", {}).get("revision")
    return rows, source_info


def _permutation_order(length: int, name: str) -> list[int]:
    identity = list(range(length))
    if name == "identity":
        return identity
    if name == "rotate1":
        return identity[1:] + identity[:1]
    if name == "rotate2":
        return identity[2:] + identity[:2]
    raise ValueError(f"unknown permutation {name!r}; expected identity, rotate1, or rotate2")


def _expanded_rows(base_rows: list[dict[str, Any]], permutations: tuple[str, ...]) -> list[dict[str, Any]]:
    """Expand rows using the exact IDs and orders from benchmarks/decisions/metrics.js."""
    if not permutations:
        raise ValueError("at least one permutation is required")
    output: list[dict[str, Any]] = []
    for source in base_rows:
        for permutation in permutations:
            order = _permutation_order(len(source["options"]), permutation)
            options = [copy.deepcopy(source["options"][index]) for index in order]
            row = copy.deepcopy(source)
            row["id"] = f"{source['semantic_id']}::perm:{permutation}"
            row["base_id"] = source["semantic_id"]
            row["permutation_id"] = permutation
            row["permutation"] = permutation
            row["cycle_shift"] = None
            row["options"] = options
            row["gold_index"] = next(index for index, option in enumerate(options) if option["id"] == source["gold_id"])
            output.append(row)
    return output


def _dataset_manifest(
    rows: list[dict[str, Any]],
    permutations: tuple[str, ...],
    source_info: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    manifest_rows = [
        {
            "id": row["id"],
            "base_id": row["base_id"],
            "semantic_id": row["semantic_id"],
            "group_id": row["group_id"],
            "case_id": row["case_id"],
            "question_id": row["question_id"],
            "family": row["family"],
            "permutation": row["permutation"],
            "gold_id": row["gold_id"],
            "gold_index": row["gold_index"],
            "state": row["state"],
            "question": row["question"],
            "option_ids": [option["id"] for option in row["options"]],
            "option_descriptions": [option["description"] for option in row["options"]],
        }
        for row in rows
    ]
    encoded = json.dumps(manifest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = _sha256_text(encoded)
    return (
        {
            "source": source_info.get("relative_path") or source_info["path"],
            "source_file": source_info,
            "eligible_semantic_decisions": len({row["semantic_id"] for row in rows}),
            "permutations": list(permutations),
            "rows": len(rows),
            "manifest_sha256": digest,
            "rows_manifest": manifest_rows,
        },
        digest,
    )


def _raw_completion_prompt(row: dict[str, Any]) -> str:
    """Build the documented no-chat-template Gemma base prompt body."""
    state = row["state"]
    evidence = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    option_lines = "\n".join(
        f"{LETTERS[index]}: {option['description']}"
        for index, option in enumerate(row["options"])
    )
    return (
        "Evidence:\n"
        f"{evidence}\n\n"
        "Criterion:\n"
        f"{row['question']}\n\n"
        "Options:\n"
        f"{option_lines}\n\n"
        "Answer:"
    )


def _validate_answer_boundary(
    tokenizer: Any,
    prompt: str,
    letters: str,
) -> tuple[list[int], list[int]]:
    """Require exact one-token labels and an unchanged prompt/label boundary."""
    input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not input_ids:
        raise ValueError("Prompt tokenization is empty")
    answer_ids: list[int] = []
    for letter in letters:
        encoded = list(tokenizer.encode(letter, add_special_tokens=False))
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        if list(tokenizer.encode(prompt + letter, add_special_tokens=False)) != input_ids + [encoded[0]]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
        answer_ids.append(encoded[0])
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Answer-slot tokens collide")
    return input_ids, answer_ids


def _encode_base_prompt(tokenizer: Any, row: dict[str, Any], max_tokens: int) -> tuple[str, list[int], list[int], str]:
    """Encode raw completion with one explicit BOS and tokenizer-specific Answer whitespace."""
    bos_token = getattr(tokenizer, "bos_token", None)
    if not isinstance(bos_token, str) or not bos_token:
        raise ValueError("Gemma base tokenizer has no usable bos_token")
    bos_ids = list(tokenizer.encode(bos_token, add_special_tokens=False))
    if len(bos_ids) != 1 or tokenizer.decode(bos_ids) != bos_token:
        raise ValueError("Gemma base tokenizer bos_token is not one exact round-trip token")
    body = bos_token + _raw_completion_prompt(row)
    last_error: Exception | None = None
    for suffix in (" ", "", "\n"):
        prompt = body + suffix
        try:
            input_ids, answer_ids = _validate_answer_boundary(tokenizer, prompt, LETTERS[: len(row["options"])])
        except ValueError as error:
            last_error = error
            continue
        if len(input_ids) > max_tokens:
            raise ValueError(
                f"Row {row['id']}: {len(input_ids)} input tokens exceed limit {max_tokens}; no truncation allowed"
            )
        if input_ids[0] != bos_ids[0] or input_ids.count(bos_ids[0]) != 1:
            raise ValueError(f"Row {row['id']}: base prompt must contain exactly one tokenizer BOS token")
        return prompt, input_ids, answer_ids, suffix
    raise ValueError(f"Row {row['id']}: no valid Answer: suffix boundary ({last_error})")


def _encode_it_prompt(helper: ModuleType, tokenizer: Any, row: dict[str, Any], max_tokens: int) -> tuple[str, list[int], list[int], None]:
    messages = helper.direct_messages(row)
    expected_options = [
        {"letter": LETTERS[index], "description": option["description"]}
        for index, option in enumerate(row["options"])
    ]
    content = messages[-1].get("content") if messages else None
    if not isinstance(content, str):
        raise TypeError(f"Row {row['id']}: Gemma IT user message has no content")
    payload = json.loads(content)
    if payload.get("options") != expected_options:
        raise ValueError(f"Row {row['id']}: Gemma IT option rendering differs from the typed-choice contract")
    prompt, input_ids, answer_ids = helper._prompt(tokenizer, row, max_tokens)
    if content not in prompt:
        raise ValueError(f"Row {row['id']}: encoded Gemma IT prompt does not contain typed-choice option text")
    return prompt, input_ids, answer_ids, None


def _forward_scores(
    model: Any,
    input_ids: list[int],
    answer_ids: list[int],
    device: str,
    dtype: Any,
    torch: Any,
) -> tuple[list[float], list[float]]:
    """Run the same text-only/P CPU-PLE forward as golden_gemma._score."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    text_model = model.model
    with torch.inference_mode():
        if device.startswith("cuda") and getattr(text_model, "hidden_size_per_layer_input", 0):
            inputs_embeds = text_model.embed_tokens(ids)
            per_layer_inputs = text_model.get_per_layer_inputs(ids.cpu(), inputs_embeds.detach().cpu())
            per_layer_inputs = per_layer_inputs.to(device=device, dtype=dtype, non_blocking=True)
            output = text_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        else:
            output = text_model(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[:, -1, :]
        answer_index = torch.tensor(answer_ids, dtype=torch.long, device=model.lm_head.weight.device)
        selected_weights = model.lm_head.weight.index_select(0, answer_index)
        logits = torch.nn.functional.linear(hidden, selected_weights)[0]
        softcap = getattr(model.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        selected = logits.float().cpu()
    if not bool(torch.isfinite(selected).all()):
        raise RuntimeError("Gemma produced non-finite answer logits")
    probabilities = torch.softmax(selected, dim=-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("Gemma produced non-finite answer probabilities")
    return selected.tolist(), probabilities.tolist()


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed_score(
    model: Any,
    tokenizer: Any,
    helper: ModuleType,
    row: dict[str, Any],
    protocol: str,
    max_tokens: int,
    device: str,
    dtype: Any,
    torch: Any,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    encode_start = time.perf_counter()
    if protocol == BASE_PROTOCOL:
        prompt, input_ids, answer_ids, answer_suffix = _encode_base_prompt(tokenizer, row, max_tokens)
    elif protocol == IT_PROTOCOL:
        prompt, input_ids, answer_ids, answer_suffix = _encode_it_prompt(helper, tokenizer, row, max_tokens)
    else:
        raise ValueError(f"Unknown prompt protocol {protocol!r}")
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    for _ in range(warmup):
        _forward_scores(model, input_ids, answer_ids, device, dtype, torch)
    _synchronize(torch, device)
    timings: list[float] = []
    scores: list[float] = []
    probabilities: list[float] = []
    for _ in range(runs):
        _synchronize(torch, device)
        start = time.perf_counter()
        current_scores, current_probabilities = _forward_scores(model, input_ids, answer_ids, device, dtype, torch)
        _synchronize(torch, device)
        timings.append((time.perf_counter() - start) * 1000.0)
        scores, probabilities = current_scores, current_probabilities

    chosen_index = max(range(len(scores)), key=scores.__getitem__)
    selected_option = row["options"][chosen_index]
    output: dict[str, Any] = {
        "id": row["id"],
        "base_id": row["base_id"],
        "semantic_id": row["semantic_id"],
        "group_id": row["group_id"],
        "case_id": row["case_id"],
        "question_id": row["question_id"],
        "family": row["family"],
        "permutation": row["permutation"],
        "cycle_shift": row["cycle_shift"],
        "state": row["state"],
        "question": row["question"],
        "options": [
            {"letter": LETTERS[index], "id": option["id"], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
        "input_ids": input_ids,
        "input_ids_sha256": _sha256_text(json.dumps(input_ids, separators=(",", ":"))),
        "answer_token_ids": answer_ids,
        "scores": scores,
        "probabilities": probabilities,
        "chosen_letter": LETTERS[chosen_index],
        "chosen_option_id": selected_option["id"],
        "chosen_option_index": chosen_index,
        "gold_option_id": row["gold_id"],
        "gold_option_index": row["gold_index"],
        "correct": selected_option["id"] == row["gold_id"],
        "timing_ms": {
            "prompt_encode": encode_ms,
            "forward_samples": timings,
            "forward_median": statistics.median(timings),
        },
    }
    if answer_suffix is not None:
        output["answer_suffix"] = answer_suffix
    return output


def _device_metadata(torch: Any, device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device_argument": device,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if device.startswith("cuda"):
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_compute_capability": [properties.major, properties.minor],
                "cuda_total_memory": properties.total_memory,
            }
        )
    return result


def _config_metadata(config: Any) -> dict[str, Any]:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    fields = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
    )
    metadata: dict[str, Any] = {}
    config_attr_errors = (AttributeError, RuntimeError, TypeError, ValueError)
    for name in fields:
        try:
            metadata[name] = getattr(text, name, None)
        except config_attr_errors:
            # Gemma 4 deliberately rejects global access to heterogeneous
            # per-layer fields such as head_dim.  Keep metadata collection
            # observational and report those values below from each layer.
            metadata[name] = None
    try:
        per_layer = list(getattr(text, "per_layer_config", ()) or ())
    except config_attr_errors:
        per_layer = []
    if per_layer:
        per_layer_metadata: dict[str, list[Any]] = {}
        for name in ("head_dim", "num_attention_heads", "num_key_value_heads", "intermediate_size"):
            values = []
            for layer_config in per_layer:
                try:
                    value = getattr(layer_config, name)
                except config_attr_errors:
                    value = None
                if value is not None and value not in values:
                    values.append(value)
            if values:
                per_layer_metadata[name] = values
        metadata["per_layer_count"] = len(per_layer)
        metadata["per_layer_values"] = per_layer_metadata
    return metadata


def _snapshot_revision(source: str) -> str | None:
    path = Path(source).expanduser()
    if not path.exists():
        return None
    for part in path.resolve().parts:
        if HEX_SHA.fullmatch(part):
            return part
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and HEX_SHA.fullmatch(parts[index + 1]):
            return parts[index + 1]
    return None


def _resolve_variant(label: str, source: str, revision: str | None) -> tuple[str, str, bool]:
    local = Path(source).expanduser().exists()
    if local:
        resolved_revision = revision or _snapshot_revision(source)
        if not resolved_revision or not HEX_SHA.fullmatch(resolved_revision):
            raise ValueError(
                f"{label}: local source needs a 40-character snapshot SHA in its path or --{label}-revision"
            )
        return str(Path(source).expanduser().resolve()), resolved_revision, True
    resolved_revision = revision or MODEL_VARIANTS[label]["revision"]
    if not HEX_SHA.fullmatch(resolved_revision):
        raise ValueError(f"{label}: remote models require a pinned 40-character revision")
    return source, resolved_revision, False


def _variant_parser_args(parser: argparse.ArgumentParser) -> None:
    for label, spec in MODEL_VARIANTS.items():
        key = label.replace("-", "_")
        parser.add_argument(
            f"--{label}",
            dest=f"{key}_source",
            default=spec["repo"],
            help=f"{label} repository or local snapshot (default: {spec['repo']})",
        )
        parser.add_argument(
            f"--{label}-revision",
            dest=f"{key}_revision",
            default=None,
            help=f"pinned 40-character revision for {label}; required for unversioned local paths",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemma-helper",
        type=Path,
        default=DEFAULT_GEMMA_HELPER,
        help="golden_gemma.py providing the shared loader and PLE path",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="frozen JSONL semantic holdout (default: benchmarks/decisions/kevala-authored36.jsonl)",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "gemma-base-vs-it.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--variants", nargs="+", choices=tuple(MODEL_VARIANTS), default=list(MODEL_VARIANTS))
    parser.add_argument(
        "--permutations",
        nargs="+",
        choices=DEFAULT_PERMUTATIONS,
        default=list(DEFAULT_PERMUTATIONS),
        help="option orders, matching benchmarks/decisions/metrics.js",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="deprecated alias selecting the first N default permutations",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=108,
        help="fail if the generated row count differs; use 0 to accept any count",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--plan", action="store_true", help="write only the dataset/protocol plan; no model or GPU work")
    _variant_parser_args(parser)
    return parser


def _validate_args(args: argparse.Namespace, rows: list[dict[str, Any]], permutations: tuple[str, ...]) -> None:
    if not permutations:
        raise ValueError("at least one permutation is required")
    if len(set(permutations)) != len(permutations):
        raise ValueError("permutations must be unique")
    if args.cycles is not None and not 1 <= args.cycles <= len(DEFAULT_PERMUTATIONS):
        raise ValueError(f"--cycles must be between 1 and {len(DEFAULT_PERMUTATIONS)}")
    if args.expected_rows < 0:
        raise ValueError("--expected-rows must be nonnegative")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.expected_rows and len(rows) != args.expected_rows:
        raise ValueError(
            f"generated {len(rows)} rows from {len({row['semantic_id'] for row in rows})} semantic decisions "
            f"and {len(permutations)} permutations; expected {args.expected_rows}; use --expected-rows 0 to override"
        )


def _plan_metadata(
    args: argparse.Namespace,
    helper: ModuleType | None,
    dataset: dict[str, Any],
    dataset_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": "gemma-decision-bench-v1",
        "script_version": SCRIPT_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": {
            "name": "Gemma 4 base versus IT decision protocol and weights ablation",
            "comparison_axis": "matched semantic rows; base/raw-completion versus IT/native-chat weights",
            "inference_scope": "reference text-only logits over declared answer tokens",
            "dtype": args.dtype,
            "device": args.device,
            "quantization": "none",
            "webgpu_timing_comparison": "not apples-to-apples and intentionally excluded",
            "semif_method_revision": SEMIF_METHOD_REVISION,
            "gemma_architecture_reference_commit": getattr(helper, "REFERENCE_SOURCE_COMMIT", None),
        },
        "option_rendering": OPTION_RENDERING,
        "dataset": dataset,
        "dataset_digest": dataset_digest,
        "protocols": {
            BASE_PROTOCOL: {
                "version": BASE_PROTOCOL,
                "weight_variants": ["base"],
                "prompt": "Evidence:, Criterion:, Options: with uppercase letter labels and Kev typed-choice descriptions; Answer: prefix",
                "chat_template": False,
                "bos": {
                    "source": "tokenizer.bos_token",
                    "placement": "explicitly prepended exactly once before Evidence:",
                    "encode_add_special_tokens": False,
                },
                "answer_boundary": "validate one token per label and exact prompt-plus-label suffix; try space, empty, newline",
            },
            IT_PROTOCOL: {
                "weight_variants": ["it"],
                "prompt": "golden_gemma._prompt and native tokenizer chat template with Kev typed-choice descriptions",
                "chat_template": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "answer_boundary": "golden_gemma._prompt exact one-token boundary validation",
            },
        },
        "sources": {
            label: {
                "repo": spec["repo"],
                "revision": spec["revision"],
                "hf_metadata_url": f"https://huggingface.co/api/models/{spec['repo']}",
                "hf_metadata_sha_field": "sha",
            }
            for label, spec in MODEL_VARIANTS.items()
        },
        "reference_helper": {
            "path": str(Path(args.gemma_helper).expanduser().resolve()),
            "git_revision": _git_revision(Path(args.gemma_helper).expanduser().resolve().parents[1]),
            "architecture_reference_commit": getattr(helper, "REFERENCE_SOURCE_COMMIT", None),
        },
    }


def _variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    timings = [row["timing_ms"]["forward_median"] for row in rows]
    for row in rows:
        groups[row["semantic_id"]].append(row["chosen_option_id"])
    unstable = {key: values for key, values in groups.items() if len(set(values)) > 1}
    correct = sum(row["correct"] for row in rows)
    return {
        "rows": len(rows),
        "semantic_decisions": len(groups),
        "permutations": len(rows) // len(groups) if groups else 0,
        "correct_rows": correct,
        "accuracy": correct / len(rows) if rows else None,
        "permutation_invariant_decisions": len(groups) - len(unstable),
        "permutation_variant_choices": unstable,
        "forward_median_ms": statistics.median(timings) if timings else None,
        "forward_p95_ms": sorted(timings)[max(0, math.ceil(len(timings) * 0.95) - 1)] if timings else None,
    }


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        helper_path = args.gemma_helper.expanduser()
        helper = None if args.plan and not helper_path.is_file() else _load_gemma_helper(helper_path)
        base_rows, source_info = _canonical_rows(args.dataset)
        permutations = tuple(DEFAULT_PERMUTATIONS[: args.cycles]) if args.cycles is not None else tuple(args.permutations)
        rows = _expanded_rows(base_rows, permutations)
        _validate_args(args, rows, permutations)
        dataset, dataset_digest = _dataset_manifest(rows, permutations, source_info)
        output = _plan_metadata(args, helper, dataset, dataset_digest)
        if args.plan:
            output["plan_only"] = True
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote plan {args.output} ({len(rows)} rows, digest {dataset_digest})")
            return

        if args.device == "cuda" and args.dtype != "bfloat16":
            print("warning: requested CUDA dtype is not BF16; this is outside the standard comparison protocol", file=sys.stderr)

        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requires torch.cuda.is_available()")
        output["environment"] = _device_metadata(torch, args.device)
        output["environment"]["transformers_version"] = None
        output["variants"] = {}

        for label in args.variants:
            spec = MODEL_VARIANTS[label]
            key = label.replace("-", "_")
            source, revision, local = _resolve_variant(
                label,
                getattr(args, f"{key}_source"),
                getattr(args, f"{key}_revision"),
            )
            protocol = BASE_PROTOCOL if spec["weight_variant"] == "base" else IT_PROTOCOL
            print(f"loading {label}: {source}@{revision} ({protocol})", flush=True)
            model = tokenizer = None
            try:
                model, tokenizer, device, dtype, config, checkpoint, model_torch, transformers = helper._load_model(
                    source,
                    revision,
                    args.device,
                    args.dtype,
                    local,
                )
                output["environment"]["transformers_version"] = transformers.__version__
                scored_rows = [
                    _timed_score(
                        model,
                        tokenizer,
                        helper,
                        row,
                        protocol,
                        args.max_tokens,
                        device,
                        dtype,
                        model_torch,
                        args.warmup,
                        args.runs,
                    )
                    for row in rows
                ]
                output["variants"][label] = {
                    "family": spec["family"],
                    "weight_variant": spec["weight_variant"],
                    "source": source,
                    "revision": revision,
                    "checkpoint": str(checkpoint),
                    "local": local,
                    "protocol": protocol,
                    "dtype": args.dtype,
                    "device": device,
                    "quantization": "none",
                    "config": _config_metadata(config),
                    "rows": scored_rows,
                    "summary": _variant_summary(scored_rows),
                }
                print(f"completed {label}: {len(scored_rows)} rows", flush=True)
            finally:
                del tokenizer, model
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()

        output["comparison"] = {
            "pairs": [
                {"family": family, "base": f"{family.lower()}-base", "it": f"{family.lower()}-it"}
                for family in ("E2B", "E4B")
            ],
            "interpretation": "Compare semantic choices and score distributions within this pinned protocol; timing is reference-only and is not compared with WebGPU.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output} ({len(rows)} rows)")
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
