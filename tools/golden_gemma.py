#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.6", "transformers>=5.17", "accelerate>=1.12", "huggingface_hub", "safetensors", "numpy", "jinja2"]
# ///
"""Generate a Transformers reference for Gemma 4 direct option scoring.

This development tool loads the text trunk of the pinned regular Gemma 4 IT
checkpoint and scores only the declared ``A`` through ``P`` answer tokens.  It
does not generate answer text.  The multimodal towers are never instantiated,
which keeps the reference useful for comparing Kevala's text path.  Gemma's
per-layer embedding (PLE) table remains on the CPU when scoring on CUDA; only
the small per-token PLE activations cross the device boundary.

The prompt uses Gemma IT's chat template with ``enable_thinking=False`` and
the SemIf direct-options contract.  The loaded checkpoints are regular
``google/gemma-4-E2B-it`` and ``google/gemma-4-E4B-it`` revisions, not the
quantized variants.  The two revisions are pinned below and can be overridden
for a local reproducibility run.

Examples::

    HF_HOME=tmp/hf uv run tools/golden_gemma.py --device cuda --dtype bfloat16 \
        --model google/gemma-4-E2B-it --output tmp/golden-gemma4-e2b.json
    uv run tools/golden_gemma.py --local /models/gemma4-E2B --device cpu \
        --cases injection email_object_state --output tmp/golden-gemma4-local.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from golden_semif import (
    CASES,
    LETTERS,
    PROMPT_VERSION,
    _question_options,
    direct_messages,
    parse_case_ids,
)

MODEL_REVISIONS = {
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
}
DEFAULT_MODEL = "google/gemma-4-E2B-it"
REFERENCE_SOURCE_COMMIT = "2c4914fb939fe9de0d8e7a798af4684d552f18b4"
DEFAULT_OUTPUT = ROOT / "tmp" / "golden-gemma4.json"
GIB = 1024**3


def _dtype(name, torch):
    return torch.float32 if name == "float32" else torch.bfloat16


def _prompt(tokenizer, row: dict, max_tokens: int) -> tuple[str, list[int], list[int]]:
    """Render the exact non-thinking Gemma IT prompt and verify label boundaries."""
    prompt = tokenizer.apply_chat_template(
        direct_messages(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids or len(input_ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(input_ids)} input tokens exceed limit {max_tokens}; no truncation allowed")

    answer_ids = []
    for letter in LETTERS[: len(row["options"])]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        answer_ids.append(encoded[0])
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != input_ids + [encoded[0]]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("Answer-slot tokens collide")
    return prompt, input_ids, answer_ids


def _rows(cases):
    """Yield SemIf-compatible direct-option rows and explicit unsupported skips."""
    for case in cases:
        for question_id, question in case["questions"].items():
            instruction, option_ids, descriptions = _question_options(question)
            if not 2 <= len(option_ids) <= len(LETTERS):
                yield None, {"case": case["id"], "question": question_id, "reason": "direct options require 2-16 options"}
                continue
            row = {
                "id": f"{case['id']}::{question_id}",
                "state": case["state"],
                "question": instruction,
                "options": [
                    {"id": option_id, "description": description}
                    for option_id, description in zip(option_ids, descriptions)
                ],
            }
            yield {"row": row, "case": case, "question_id": question_id, "question": question}, None


def _available_memory() -> tuple[int | None, int | None]:
    """Return MemAvailable and SwapFree from Linux's procfs, in bytes."""
    try:
        values = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, value, unit = line.split(maxsplit=2)
                if key in {"MemAvailable:", "SwapFree:"}:
                    values[key[:-1]] = int(value) * (1024 if unit.strip() == "kB" else 1)
        return values.get("MemAvailable"), values.get("SwapFree")
    except (FileNotFoundError, OSError, ValueError):
        return None, None


def _estimate_text_bytes(config, dtype_name: str) -> int:
    """Conservatively estimate text-only parameter storage before a CPU load."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    hidden = int(text.hidden_size)
    vocab = int(text.vocab_size)
    layers = int(text.num_hidden_layers)
    intermediate = int(text.intermediate_size)
    ple_dim = int(getattr(text, "hidden_size_per_layer_input", 0) or 0)
    ple_vocab = int(getattr(text, "vocab_size_per_layer_input", 0) or 0)

    # This intentionally overestimates ordinary dense text weights.  It also
    # includes the packed PLE table, which dominates Gemma 4's memory use.
    parameters = vocab * hidden
    parameters += ple_vocab * layers * ple_dim
    parameters += hidden * layers * ple_dim
    parameters += layers * 4 * hidden * hidden
    for layer_idx in range(layers):
        double_wide = bool(getattr(text, "use_double_wide_mlp", False)) and layer_idx >= layers - int(
            getattr(text, "num_kv_shared_layers", 0)
        )
        width = intermediate * (2 if double_wide else 1)
        parameters += 3 * hidden * width

    bytes_per_parameter = 4 if dtype_name == "float32" else 2
    # A small headroom allowance covers allocator fragmentation and activations.
    return int(parameters * bytes_per_parameter * 1.20)


def _check_cpu_memory(config, dtype_name: str, device_name: str) -> None:
    if device_name != "cpu":
        return
    estimate = _estimate_text_bytes(config, dtype_name)
    if estimate <= 16 * GIB:
        return
    available, swap_free = _available_memory()
    if available is None or swap_free is None:
        raise RuntimeError(
            f"estimated CPU load is {estimate / GIB:.1f} GiB (>16 GiB), but /proc/meminfo is unavailable; "
            "refusing an unbounded load"
        )
    if available < 32 * GIB or swap_free < 4 * GIB:
        raise RuntimeError(
            f"estimated CPU load is {estimate / GIB:.1f} GiB (>16 GiB), requiring MemAvailable>=32 GiB and "
            f"SwapFree>=4 GiB; found {available / GIB:.1f} GiB and {swap_free / GIB:.1f} GiB"
        )
    print(
        f"CPU memory preflight: estimated {estimate / GIB:.1f} GiB, "
        f"MemAvailable {available / GIB:.1f} GiB, SwapFree {swap_free / GIB:.1f} GiB",
        file=sys.stderr,
    )


def _checkpoint_path(source: str, revision: str | None, local: bool) -> Path:
    if local:
        path = Path(source) / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"local Gemma checkpoint not found: {path}")
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=source, filename="model.safetensors", revision=revision))


def _target_key(source_key: str) -> str | None:
    """Map conditional-generation checkpoint names onto the text-only model."""
    if source_key.startswith("model.language_model."):
        return "model." + source_key[len("model.language_model.") :]
    if source_key.startswith("language_model."):
        return "model." + source_key[len("language_model.") :]
    if source_key.startswith(("model.", "lm_head.")):
        return source_key
    return None


def _load_model(source: str, revision: str | None, device_name: str, dtype_name: str, local: bool):
    import torch
    import transformers
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requires torch.cuda.is_available()")
    device = "cuda:0" if device_name == "cuda" else "cpu"
    dtype = _dtype(dtype_name, torch)
    common = {"local_files_only": local, "trust_remote_code": False}
    if not local:
        common["revision"] = revision
    config = transformers.AutoConfig.from_pretrained(source, **common)
    model_type = getattr(config, "model_type", "")
    if model_type != "gemma4":
        raise ValueError(f"Gemma 4 reference expects model_type='gemma4', got {model_type!r}")
    _check_cpu_memory(config, dtype_name, device_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    checkpoint = _checkpoint_path(source, revision, local)
    text_config = config.get_text_config()

    # Construct only Gemma4ForCausalLM under meta tensors.  Loading the full
    # conditional-generation model would allocate the unused vision/audio
    # towers and prevents the E4B PLE table from being managed independently.
    cls = getattr(transformers, "Gemma4ForCausalLM", None)
    if cls is None:
        raise RuntimeError("Installed transformers lacks Gemma4ForCausalLM")
    with init_empty_weights():
        model = cls(text_config)
    target_parameters = dict(model.named_parameters())
    target_buffers = dict(model.named_buffers())
    persistent_buffers = set(target_buffers) & set(model.state_dict())
    # Persistent tensors such as Gemma's learned layer scalars are stored as
    # buffers rather than parameters.  Include both namespaces so checkpoint
    # values replace the generated defaults before placement.
    target_names = set(target_parameters) | set(target_buffers)
    loaded = set()
    ple_prefix = "model.embed_tokens_per_layer."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
        for source_key in reader.keys():  # noqa: SIM118
            target_key = _target_key(source_key)
            if target_key is None or target_key not in target_names:
                continue
            value = reader.get_tensor(source_key)
            if value.dtype != dtype:
                value = value.to(dtype=dtype)
            target_device = "cpu" if target_key.startswith(ple_prefix) else device
            set_module_tensor_to_device(
                model,
                target_key,
                target_device,
                value=value,
                dtype=dtype,
                clear_cache=False,
            )
            loaded.add(target_key)

    missing_buffers = sorted(persistent_buffers - loaded)
    if missing_buffers:
        raise RuntimeError(f"Gemma checkpoint is missing persistent buffers: {missing_buffers}")

    # The official checkpoints tie the LM head to the main token embedding and
    # may omit the duplicate tensor.  Tie after loading so a missing duplicate
    # does not leave a meta parameter behind.
    model.tie_weights()

    # ``init_empty_weights`` leaves generated buffers (for example
    # ``layer_scalar`` and rotary inverse frequencies) on the host.  The
    # checkpoint contains some of these buffers, but they still need the same
    # placement as their decoder layer before a CUDA forward.  Keep only the
    # packed PLE embedding and its scale on the CPU: ``_score`` performs that
    # lookup explicitly and transfers the small per-token result.
    for buffer_name, buffer in list(model.named_buffers()):
        if buffer is None:
            continue
        buffer_device = "cpu" if buffer_name.startswith(ple_prefix) else device
        if str(buffer.device) != buffer_device:
            set_module_tensor_to_device(
                model,
                buffer_name,
                buffer_device,
                value=buffer,
                clear_cache=False,
            )
    missing = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(f"text-only Gemma checkpoint is missing {len(missing)} parameters: {preview}")
    if not loaded:
        raise RuntimeError(f"no text-only Gemma weights found in {checkpoint}")
    model.eval()
    model.config.use_cache = False
    gc.collect()
    return model, tokenizer, device, dtype, config, checkpoint, torch, transformers


class _Trace:
    def __init__(self, model, path: str | None, layer_specs: str):
        self.path = Path(path).resolve() if path else None
        self.active = bool(path)
        self.values: dict[str, object] = {}
        self.hooks = []
        if not self.active:
            return
        layers = model.model.layers
        selected = set()
        for item in layer_specs.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item == "last":
                selected.add(len(layers) - 1)
            else:
                index = int(item)
                if not 0 <= index < len(layers):
                    raise ValueError(f"trace layer {index} is outside 0..{len(layers) - 1}")
                selected.add(index)

        def capture(name):
            def hook(_module, _inputs, output):
                if not self.active or name in self.values:
                    return
                value = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(value, "last_hidden_state"):
                    value = value.last_hidden_state
                self.values[name] = value.detach().float().cpu()

            return hook

        self.hooks.append(model.model.embed_tokens_per_layer.register_forward_hook(capture("ple_embed")))
        for index in sorted(selected):
            self.hooks.append(model.model.layers[index].register_forward_hook(capture(f"layer_{index}")))

    def finish(self, prompt: str, input_ids: list[int], final_hidden, metadata: dict) -> dict | None:
        if not self.active:
            return None
        self.active = False
        self.path.mkdir(parents=True, exist_ok=True)
        files = {}
        values = {**self.values, "final_hidden": final_hidden.detach().float().cpu()}
        for name, value in values.items():
            target = self.path / f"{name}.f32"
            value.numpy().astype("<f4").tofile(target)
            files[name] = {"path": str(target), "shape": list(value.shape), "dtype": "float32"}
        descriptor = {
            **metadata,
            "prompt": prompt,
            "input_ids": input_ids,
            "files": files,
        }
        with (self.path / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(descriptor, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for hook in self.hooks:
            hook.remove()
        return descriptor


def _score(model, tokenizer, row, max_tokens, device, dtype, torch, trace):
    prompt, input_ids, answer_ids = _prompt(tokenizer, row, max_tokens)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    text_model = model.model
    with torch.inference_mode():
        # PLE's packed table is intentionally CPU resident on CUDA.  Look it
        # up there, then move only the raw [batch, sequence, layers, ple_dim]
        # tensor to the accelerator.  Gemma4TextModel.forward performs the
        # context projection itself, so projecting here would apply it twice.
        if device.startswith("cuda") and text_model.hidden_size_per_layer_input:
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
        last_hidden = hidden[0].float().cpu()
    probabilities = torch.softmax(selected, dim=-1).tolist()
    return prompt, input_ids, answer_ids, selected.tolist(), probabilities, last_hidden


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="pinned regular Gemma 4 IT repository")
    ap.add_argument("--revision", help="40-character Hub revision; defaults to the pinned Gemma revision")
    ap.add_argument("--local", metavar="MODEL_DIR", help="load tokenizer/config/model.safetensors locally")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="golden JSON destination")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--cases", nargs="+", help="case IDs, as separate values or comma-separated values")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-activations", action="store_true", help="skip final hidden and trace activation dumps")
    ap.add_argument("--trace", metavar="PATH", help="dump PLE/layer/final activations for the first scored case")
    ap.add_argument("--trace-layers", default="0,last", help="comma-separated decoder layer indices, plus 'last'")
    a = ap.parse_args()
    if a.max_tokens < 1:
        ap.error("--max-tokens must be positive")
    if a.no_activations and a.trace:
        ap.error("--trace cannot be combined with --no-activations")
    try:
        wanted = parse_case_ids(a.cases)
    except ValueError as error:
        ap.error(str(error))
    cases = [case for case in CASES if wanted is None or case["id"] in wanted]
    if not cases:
        ap.error("--cases selected no cases")

    local = a.local is not None
    source = os.path.abspath(a.local) if local else a.model
    revision = a.revision or ("local" if local else MODEL_REVISIONS.get(a.model))
    if not local and not revision:
        ap.error(f"no pinned revision is known for {a.model!r}; pass --revision")
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision):
        ap.error("remote models require a pinned 40-character --revision")
    try:
        model, tokenizer, device, dtype, config, checkpoint, torch, transformers = _load_model(
            source, revision, a.device, a.dtype, local
        )
        trace = _Trace(model, None if a.no_activations else a.trace, a.trace_layers)
    except (OSError, RuntimeError, ValueError) as error:
        ap.error(str(error))

    labels = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            ap.error(f"label {letter!r} is not one exact tokenizer token")
        labels.append(encoded[0])
    if len(labels) != len(set(labels)):
        ap.error("label token IDs collide")
    if max(labels) >= model.config.vocab_size:
        ap.error("label token ID exceeds the LM vocabulary")

    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    metadata = {
        "model": {
            "source": a.model,
            "revision": revision,
            "checkpoint": checkpoint.name,
            "dtype": a.dtype,
            "device": device,
            "hidden_size": int(model.config.hidden_size),
            "vocab_size": int(model.config.vocab_size),
            "text_only": True,
            "architecture_reference_commit": REFERENCE_SOURCE_COMMIT,
            "implementation_sha256": hashlib.sha256(
                Path(sys.modules[type(model).__module__].__file__).read_bytes()
            ).hexdigest(),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        "prompt_version": PROMPT_VERSION,
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
        "label_contract": {"letters": list(LETTERS), "token_ids": labels, "max_options": len(LETTERS)},
        "readout": "native last-position logits restricted to declared answer slots; no generated tokens",
        "final_logit_softcapping": getattr(text_config, "final_logit_softcapping", None),
        "probability_status": "conditional option score; uncalibrated as decision confidence",
        "load_policy": "Gemma4ForCausalLM text trunk; monolithic safetensors; PLE embedding CPU-resident on CUDA",
        "kv_sharing": "native Gemma4TextModel shared_kv_states path",
    }
    output_cases = []
    skipped = []
    first_hidden = None
    first_hidden_meta = None
    first_trace_meta = None
    for item, skip in _rows(cases):
        if skip:
            skipped.append(skip)
            continue
        row = item["row"]
        prompt, input_ids, answer_ids, scores, probabilities, hidden = _score(
            model, tokenizer, row, a.max_tokens, device, dtype, torch, trace
        )
        case = item["case"]
        question_id = item["question_id"]
        output_cases.append(
            {
                "id": row["id"],
                "case_id": case["id"],
                "question_id": question_id,
                "state": case["state"],
                "questions": {question_id: item["question"]},
                "options": row["options"],
                "input_ids": input_ids,
                "answer_token_ids": answer_ids,
                "scores": [scores],
                "logits": [scores],
                "probs": [probabilities],
                "input_tokens": len(input_ids),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_ids_sha256": hashlib.sha256(
                    b"".join(int(value).to_bytes(4, "little") for value in input_ids)
                ).hexdigest(),
                "prompt_version": PROMPT_VERSION,
                "rendered_prompt": prompt,
            }
        )
        if first_hidden is None and not a.no_activations:
            first_hidden = hidden
            first_hidden_meta = {"case_id": case["id"], "question_id": question_id, "input_ids": input_ids}
            if a.trace:
                first_trace_meta = trace.finish(prompt, input_ids, hidden, first_hidden_meta)
        print(f"{row['id']:<30} tokens={len(input_ids)}  {[f'{value:.4f}' for value in probabilities]}")

    result = {**metadata, "cases": output_cases, "skipped": skipped}
    if first_hidden is not None and not a.trace:
        activation_path = Path(a.output).resolve().with_suffix(".final.f32")
        activation_path.parent.mkdir(parents=True, exist_ok=True)
        first_hidden.numpy().astype("<f4").tofile(activation_path)
        result["activation_dump"] = {
            **first_hidden_meta,
            "path": str(activation_path),
            "shape": list(first_hidden.shape),
            "dtype": "float32",
        }
    if first_trace_meta is not None:
        result["trace"] = first_trace_meta
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
        stream.write("\n")
    print(f"wrote {output} ({len(output_cases)} rows, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
