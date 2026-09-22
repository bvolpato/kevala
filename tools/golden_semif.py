#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.6", "transformers>=5.17", "numpy", "accelerate"]
# ///
"""Reference direct-label scorer for the SemIf contract.

This is a development tool only.  It deliberately evaluates one question at a
time and records the exact tokenizer prompt, label token IDs, native
last-position logits, and the conditional distribution over the declared
options.  It never generates answer text and it does not load or claim a
SemIf fine-tune: the model is the pinned upstream Qwen checkpoint passed on
the command line.

The prompt and option contract follows SemIf's MIT-licensed
``direct-options-v1`` implementation:

* the system instruction is fixed;
* the user payload is ``json.dumps(..., ensure_ascii=False)`` with Python's
  default separators and insertion order;
* Qwen's pinned tokenizer renders the chat template with
  ``add_generation_prompt=True`` and ``enable_thinking=False``; and
* labels are exact single uppercase tokens from A through P.

Examples:

  HF_HOME=tmp/hf uv run tools/golden_semif.py --device cuda --dtype bfloat16 \
      --output tmp/golden-semif-qwen35-4b.json
  HF_HOME=tmp/hf uv run tools/golden_semif.py --local /models/qwen35-4b \
      --revision local-qwen35-4b --cases injection email_object_state
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

from golden_kev import CASES, parse_case_ids  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "golden-semif.json"
PROMPT_VERSION = "direct-options-v1"
LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def _render(value, indent: int = 0) -> str:
    """Match Kev's ``kev.api.render`` for option descriptions and instructions."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {_render(item, indent + 1).lstrip()}" for item in value)
    if isinstance(value, dict):
        return "\n".join(
            f"{pad}{key}:\n{_render(item, indent + 1)}"
            if isinstance(item, (dict, list))
            else f"{pad}{key}: {_render(item)}"
            for key, item in value.items()
        )
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _option_text(name: str, description) -> str:
    return name if description is None or description == "" else f"{name}: {_render(description)}"


def _py_str(value) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    return str(value)


def _question_options(question: dict) -> tuple[str, list[str], list[str]]:
    """Return (instruction, option IDs, rendered descriptions) in Kev order.

    This mirrors the option text in Kev's ``kev.api.to_record`` and Rust
    ``parse_questions``.  Score descriptions are the raw rendered criteria,
    matching the upstream API contract.
    """
    question_type = question.get("type")
    instruction = _render(question.get("instructions"))
    criteria = question.get("criteria")
    if question_type == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        ids = ["false", "true"]
        descriptions = [_option_text("no", criteria.get("false")), _option_text("yes", criteria.get("true"))]
        return instruction, ids, descriptions
    if question_type == "choice":
        if isinstance(criteria, dict):
            ids = [str(key) for key in criteria]
            descriptions = [_option_text(key, criteria[key]) for key in criteria]
            return instruction, ids, descriptions
        if isinstance(criteria, list):
            ids, descriptions = [], []
            for value in criteria:
                key = _py_str(value)
                if key not in ids:
                    ids.append(key)
                    descriptions.append(key)
            return instruction, ids, descriptions
        raise ValueError("choice criteria must be an object or list")
    if question_type == "score":
        if isinstance(criteria, list):
            values = criteria
        elif isinstance(criteria, dict):
            values = list(criteria)
        elif isinstance(criteria, str):
            values = list(criteria)
        else:
            raise ValueError("score criteria must be a list")
        return instruction, [str(index) for index in range(len(values))], [_render(value) for value in values]
    raise ValueError(f"unknown question type {question_type!r}")


def direct_messages(row: dict) -> list[dict]:
    """Build the exact SemIf messages, preserving JSON values and key order."""
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    question = row["question"]
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a nonempty string")
    options = row["options"]
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError("SemIf direct options require 2-16 options")
    payload = {
        "evidence": state,
        "criterion": question,
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(options)
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[str, list[int], list[int]]:
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


def _state_prefix(tokenizer, state) -> list[int]:
    """Build SemIf's cache prefix, ending before the punctuation boundary token."""
    row = {
        "id": "prefix-only",
        "state": state,
        "question": "prefix boundary placeholder",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
    }
    turns = direct_messages(row)
    prompt = tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    payload = turns[-1]["content"]
    if prompt.count(payload) != 1:
        raise ValueError("Cannot locate the unmodified evidence payload in the chat template")
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
    if not payload.startswith(evidence):
        raise ValueError("Evidence serialization changed")
    prefix_text = prompt[: prompt.index(payload)] + evidence
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    if not prefix:
        raise ValueError("Empty cache prefix")
    return prefix[:-1]


def _load_model(source: str, revision: str | None, device_name: str, dtype_name: str, local: bool):
    import torch
    import transformers

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requires torch.cuda.is_available()")
    device = "cuda:0" if device_name == "cuda" else "cpu"
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    common = {
        "revision": None if local else revision,
        "local_files_only": local,
        "trust_remote_code": False,
    }
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    kwargs = {
        **common,
        "dtype": dtype,
        "device_map": {"": device},
        "low_cpu_mem_usage": True,
    }

    # Qwen3.5 may be published as a text-only checkpoint or as the
    # conditional-generation wrapper.  In both cases run only its text model
    # and apply the selected LM-head rows ourselves.  The wrapper's visual
    # module is never called and is released before scoring.
    model_type = getattr(config, "model_type", "")
    wrapper = None
    if model_type == "qwen3_5_text":
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks Qwen3_5ForCausalLM")
        wrapper = cls.from_pretrained(source, config=config, **kwargs)
        language_model = wrapper.model
        lm_head = wrapper.lm_head
    elif model_type == "qwen3_5":
        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if cls is None:
            cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks Qwen3.5 conditional-generation support")
        wrapper = cls.from_pretrained(source, **kwargs)
        container = getattr(wrapper, "model", wrapper)
        language_model = getattr(container, "language_model", None)
        if language_model is None:
            language_model = getattr(wrapper, "language_model", None)
        if language_model is None:
            raise RuntimeError("Qwen3.5 checkpoint has no language_model submodule")
        lm_head = wrapper.lm_head
    else:
        raise ValueError(f"SemIf reference expects Qwen3.5, got model_type={model_type!r}")

    language_model.eval()
    lm_head.eval()
    label_width = int(getattr(language_model.config, "hidden_size", 0))
    if label_width <= 0:
        raise RuntimeError("Could not determine Qwen3.5 hidden size")
    label_weights = lm_head.weight.detach()
    label_bias = None if getattr(lm_head, "bias", None) is None else lm_head.bias.detach()
    # The visual tower is not part of the reference forward path.  Dropping
    # the wrapper here also avoids keeping its vision parameters alive.
    del wrapper
    gc.collect()
    return language_model, label_weights, label_bias, tokenizer, device, dtype, label_width, torch, transformers


def _score(language_model, label_weights, label_bias, tokenizer, row, max_tokens, device, torch):
    prompt, input_ids, answer_ids = _encode_prompt(tokenizer, row, max_tokens)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    with torch.inference_mode():
        output = language_model(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[:, -1, :]
        selected_weights = label_weights.index_select(0, torch.tensor(answer_ids, device=label_weights.device))
        logits = torch.nn.functional.linear(hidden, selected_weights, None if label_bias is None else label_bias.index_select(0, torch.tensor(answer_ids, device=label_bias.device)))
        selected = logits[0].float().cpu()
        last_hidden = hidden[0].float().cpu()
    probabilities = torch.softmax(selected, dim=-1).tolist()
    return prompt, input_ids, answer_ids, selected.tolist(), probabilities, last_hidden


def _semif_rows(cases):
    for case in cases:
        for question_id, question in case["questions"].items():
            instruction, option_ids, descriptions = _question_options(question)
            if not 2 <= len(option_ids) <= len(LETTERS):
                yield None, {"case": case["id"], "question": question_id, "reason": "SemIf requires 2-16 options"}
                continue
            row_id = f"{case['id']}::{question_id}"
            row = {
                "id": row_id,
                "state": case["state"],
                "question": instruction,
                "options": [
                    {"id": option_id, "description": description}
                    for option_id, description in zip(option_ids, descriptions)
                ],
            }
            yield {
                "row": row,
                "case": case,
                "question_id": question_id,
                "question": question,
            }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="pinned upstream Qwen3.5 model or local model directory")
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="40-character Hub revision, or a local provenance label")
    ap.add_argument("--local", metavar="MODEL_DIR", help="load tokenizer and weights from this local directory")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="golden JSON destination")
    ap.add_argument("--labels-output", help="optional little-endian F32 selected LM-head rows destination")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--cases", nargs="+", help="case IDs, as separate values or comma-separated values")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-activations", action="store_true", help="skip the first-row final hidden activation dump")
    a = ap.parse_args()
    if a.max_tokens < 1:
        ap.error("--max-tokens must be positive")
    try:
        wanted = parse_case_ids(a.cases)
    except ValueError as error:
        ap.error(str(error))
    cases = [case for case in CASES if wanted is None or case["id"] in wanted]
    if not cases:
        ap.error("--cases selected no cases")
    local = a.local is not None
    source = os.path.abspath(a.local) if local else a.model
    if not local and not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        ap.error("remote models require a pinned 40-character --revision")
    revision = a.revision if a.revision else ("local" if local else DEFAULT_REVISION)
    try:
        language_model, label_weights, label_bias, tokenizer, device, dtype, hidden_size, torch, transformers = _load_model(
            source, revision, a.device, a.dtype, local
        )
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
    if max(labels) >= label_weights.shape[0]:
        ap.error("label token ID exceeds the LM vocabulary")
    if a.labels_output:
        labels_path = Path(a.labels_output).resolve()
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        label_weights.index_select(0, torch.tensor(labels, device=label_weights.device)).float().cpu().numpy().astype("<f4").tofile(labels_path)
    else:
        labels_path = None

    metadata = {
        "model": {
            "source": source,
            "revision": revision,
            "dtype": a.dtype,
            "device": device,
            "hidden_size": hidden_size,
            "vocab_size": int(label_weights.shape[0]),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        "prompt_version": PROMPT_VERSION,
        "label_contract": {"letters": list(LETTERS), "token_ids": labels, "max_options": len(LETTERS)},
        "readout": "native last-position logits restricted to declared answer slots; no generated tokens",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
    if labels_path:
        metadata["label_contract"].update({"rows_path": str(labels_path), "rows_dtype": "float32", "rows_shape": [len(labels), hidden_size]})
    output_cases = []
    skipped = []
    first_hidden = None
    first_hidden_meta = None
    for item, skip in _semif_rows(cases):
        if skip:
            skipped.append(skip)
            continue
        row = item["row"]
        prompt, input_ids, answer_ids, logits, probabilities, hidden = _score(
            language_model, label_weights, label_bias, tokenizer, row, a.max_tokens, device, torch
        )
        case = item["case"]
        qid = item["question_id"]
        output_cases.append({
            "id": row["id"],
            "case_id": case["id"],
            "question_id": qid,
            "state": case["state"],
            "questions": {qid: item["question"]},
            "options": row["options"],
            "input_ids": input_ids,
            "answer_token_ids": answer_ids,
            "logits": [logits],
            "probs": [probabilities],
            "input_tokens": len(input_ids),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_ids_sha256": hashlib.sha256(bytes().join(int(value).to_bytes(4, "little") for value in input_ids)).hexdigest(),
            "prompt_version": PROMPT_VERSION,
            "rendered_prompt": prompt,
            "cache_prefix_tokens": len(_state_prefix(tokenizer, row["state"])),
        })
        if first_hidden is None and not a.no_activations:
            first_hidden = hidden
            first_hidden_meta = {"case_id": case["id"], "question_id": qid, "input_ids": input_ids}
        print("%-30s tokens=%d  %s" % (row["id"], len(input_ids), [f"{value:.4f}" for value in probabilities]))

    result = {**metadata, "cases": output_cases, "skipped": skipped}
    if first_hidden is not None:
        activation_dir = ROOT / "tmp" / "golden-semif"
        activation_dir.mkdir(parents=True, exist_ok=True)
        activation_path = activation_dir / "final.f32"
        first_hidden.numpy().astype("<f4").tofile(activation_path)
        result["activation_dump"] = {**first_hidden_meta, "path": str(activation_path), "shape": list(first_hidden.shape), "dtype": "float32"}
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False)
        stream.write("\n")
    print(f"wrote {output} ({len(output_cases)} rows, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
