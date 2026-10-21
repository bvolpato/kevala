#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers>=5.17", "requests", "jinja2", "torch>=2.6", "safetensors"]
# [tool.uv]
# exclude-newer = "P7D"
# ///
"""Freeze H2O's pinned shim prompt, tokenizer, and answer postprocessing.

By default this loads no model weights. The shim and serving contract are fetched from the
pinned upstream checkpoint, and Transformers materializes its text tokenizer.
Use --tokenizer-output to save a tokenizer.json accepted by the Rust tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL = "h2oai/h2o-lightning-4b"
REVISION = "acaf0d4ea251e54de928c75ef4352670d33192d3"
SOURCES = {
    "h2o_lightning_shim.py": "91a15e3003668547f1d3ea361a80ac9e6852e3a072a9043608044efd365dc512",
    "serve_config.json": "cdd4cd628c8b0ebbeffa0fde8c739ec32f9f9726b7b2539ffb1831d608eb2488",
}

CASES = [
    ("unicode_state", {"language": "日本語", "float": 1.0, "small": 1e-7}, {
        "q": {"instructions": {"task": "日本語", "steps": [1, True]}, "criteria": {"z": None, "a": ""}},
    }),
    ("noul_alias_threshold", "  This ticket needs help.  ", {
        "q": {"type": "noul", "instructions": "  Escalate?", "criteria": {"yes": "urgent", "no": {"routine": True}}, "threshold": 0.8},
    }),
    ("noul_defaults", None, {"q": {"type": "noul", "criteria": None}}),
    ("python_falsey_noul", "", {
        "empty": {"type": "noul", "instructions": "\u001c  Question?", "criteria": []},
        "zero": {"type": "noul", "criteria": 0},
        "false": {"type": "noul", "criteria": False},
    }),
    ("one_level_score", ["sample", "café"], {"q": {"type": "score", "instructions": [], "criteria": [{"quality": "ok"}]}}),
    ("multi_questions", {"title": "Refund", "amount": 10.0}, {
        "route": {"type": "choice", "instructions": "Route it", "criteria": {"billing": "payments", "support": "technical"}},
        "refund": {"type": "noul", "instructions": "Is this a refund?"},
        "quality": {"type": "score", "instructions": "Quality", "criteria": ["low", "mid", "high"]},
    }),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/golden-h2o.json"))
    parser.add_argument("--cache", type=Path, default=Path("tmp/h2o-reference"))
    parser.add_argument("--tokenizer-output", type=Path, default=Path("tmp/h2o-reference/tokenizer"))
    parser.add_argument("--weights", type=Path, help="Load a local merged checkpoint for numerical backbone parity")
    parser.add_argument("--with-weights", action="store_true", help="Load the pinned HF checkpoint for numerical parity")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for name, digest in SOURCES.items():
        path = args.cache / name
        if not path.exists():
            response = requests.get(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            raise ValueError(f"{name}: expected SHA256 {digest}, got {got}")
    spec = importlib.util.spec_from_file_location("h2o_reference_shim", args.cache / "h2o_lightning_shim.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    serve = json.loads((args.cache / "serve_config.json").read_text())
    contract = shim.Contract(serve, env={})
    tokenizer = AutoTokenizer.from_pretrained(args.weights or args.model, revision=None if args.weights else args.revision)
    args.tokenizer_output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_output)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    probe = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": f"d{i}" for i in range(len(contract.labels))}}
    prompt = contract.question_prompt(shim.render_state({"x": 1}, contract.state_compact), probe)[0]
    base = encode(prompt)
    label_ids = []
    for label in contract.labels:
        full = encode(prompt + " " + label)
        if full[:-1] != base or len(full) != len(base) + 1 or full[-1] in label_ids:
            raise ValueError(f"label {label!r} is not one distinct answer-slot token")
        label_ids.append(full[-1])
    cases = []
    for cid, state, questions in CASES:
        prompts = []
        for qid, question in questions.items():
            text, names, descriptions, labels = contract.question_prompt(shim.render_state(state, contract.state_compact), question, qid)
            messages = [{"role": "system", "content": contract.system.strip()}, {"role": "user", "content": contract.question_parts(shim.render_state(state, contract.state_compact), question, qid)[0].strip()}]
            chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if encode(tokenizer.decode(chat_ids, skip_special_tokens=False) + contract.prefill) != encode(text):
                raise ValueError(f"{cid}/{qid}: upstream prompt differs from checkpoint chat template")
            prompts.append({"id": qid, "prompt": text, "input_ids": encode(text), "names": names, "descriptions": descriptions, "labels": labels})
        cases.append({"id": cid, "state": state, "questions": questions, "prompts": prompts})
    # Include a tokenizer-boundary truncation case with non-ASCII text.
    state = "café 日本語 hello " * 24
    ids = encode(state)
    limit = 17
    truncated = tokenizer.decode(ids[:limit // 2], skip_special_tokens=False) + "\n...\n" + tokenizer.decode(ids[-(limit // 2):], skip_special_tokens=False)
    trunc_q = {"type": "noul", "instructions": "Contains language?"}
    cases.append({"id": "truncated_state", "state": state, "questions": {"q": trunc_q}, "max_state_tokens": limit,
                  "prompts": [{"id": "q", "prompt": contract.question_prompt(truncated, trunc_q)[0], "input_ids": encode(contract.question_prompt(truncated, trunc_q)[0])}]})
    numeric = []
    for typ, logits, criteria, threshold in [
        ("noul", [0.0, 0.0], None, 0.8), ("noul", [-0.01, 0.0], None, 0.2),
        ("noul", [2.0, 0.0], None, None), ("choice", [0.0, 1.0, -1.0], {"a": "first", "b": "second", "c": "third"}, None),
        ("score", [0.0, 1.0, -1.0], ["low", {"mid": True}, "high"], None), ("score", [0.0], ["only"], None),
    ]:
        question = {"type": typ, "criteria": criteria}
        if threshold is not None:
            question["threshold"] = threshold
        names, descs = shim.options_of(question)
        answer = shim.format_answer(typ, names, descs, logits, contract.temperature_by_type[typ], contract.noul_floor, shim.noul_threshold(question))
        numeric.append({"question": question, "logits": logits, "answer": answer, "probabilities": shim.probabilities(logits, contract.temperature_by_type[typ])})
    out = {"model": args.model, "revision": args.revision, "source_sha256": SOURCES, "scope": "prompt/tokenizer and postprocessing only; no weights loaded",
           "serve_config": serve, "label_token_ids": label_ids, "cases": cases, "numeric": numeric}
    if args.weights or args.with_weights:
        import torch
        from safetensors import safe_open
        from transformers import AutoModelForImageTextToText
        from huggingface_hub import snapshot_download

        checkpoint = args.weights or Path(snapshot_download(args.model, revision=args.revision, allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
        # Read selected rows from the original checkpoint, retaining any native f32 head.
        tensors = {}
        for path in checkpoint.glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    tensors[name] = path
        head_names = [name for name in tensors if name in ("lm_head.weight", "model.lm_head.weight", "model.language_model.lm_head.weight", "language_model.lm_head.weight")]
        if len(head_names) > 1:
            raise ValueError("ambiguous native LM head")
        head_name = head_names[0] if head_names else "model.language_model.embed_tokens.weight"
        with safe_open(tensors[head_name], framework="pt", device="cpu") as source:
            rows = source.get_slice(head_name)
            labels_weight = torch.cat([rows[tid:tid+1].float() for tid in label_ids]).to(args.device)
        model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="eager").to(args.device).eval()
        trunk = model.model.language_model
        references = []
        with torch.inference_mode():
            for case in cases:
                if case["id"] not in ("unicode_state", "noul_alias_threshold", "one_level_score"):
                    continue
                prompt = case["prompts"][0]
                input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=args.device)
                hidden = trunk(input_ids=input_ids, use_cache=False).last_hidden_state[0, -1].float()
                logits = (labels_weight[:len(shim.options_of(case["questions"][prompt["id"]])[0])] @ hidden).cpu().tolist()
                question = case["questions"][prompt["id"]]
                typ = question.get("type", "choice")
                names, descs = shim.options_of(question)
                references.append({"id": case["id"], "state": case["state"], "questions": case["questions"], "input_ids": prompt["input_ids"], "logits": logits,
                                   "answer": shim.format_answer(typ, names, descs, logits, contract.temperature_by_type[typ], contract.noul_floor, shim.noul_threshold(question)),
                                   "probabilities": shim.probabilities(logits, contract.temperature_by_type[typ])})
        out["scope"] = "prompt/tokenizer, postprocessing, and native BF16 backbone with original f32 label-row projection"
        out["execution"] = {"device": args.device, "backbone_dtype": "bfloat16", "readout_dtype": "float32", "lm_head_source": head_name}
        out["weight_cases"] = references
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.output}; {out['scope']}")


if __name__ == "__main__":
    main()
