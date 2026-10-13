#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.6", "transformers>=5.17", "accelerate>=1.12", "safetensors", "huggingface_hub", "numpy", "pillow"]
# ///
"""Generate a full Clef text reference using its unmodified upstream encoder and head.

The default reference is BF16. For a bounded CPU run, stream the decoder layers
from disk with --offload-dir. The input/output embeddings and joint head stay on
CPU; the vision tower is never used. No quantization is applied to the reference.

uv run tools/golden_clef.py --local /models/clef-flash \
    --upstream /models/clef-flash/joint_schema_model.py \
    --offload-dir tmp/clef-reference-offload --output tmp/golden-clef.json
kevala parity-clef tmp/clef-flash-q8.kevala tmp/golden-clef.json --max-dp 0.03
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from golden_kev import CASES, parse_case_ids

MODEL = "Cloudflare/clef-flash"
REVISION = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
UPSTREAM_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def source_module(path: Path, revision: str):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if revision == REVISION and digest != UPSTREAM_SHA256:
        raise ValueError(f"upstream source SHA256 differs from pinned Clef revision: {digest}")
    spec = importlib.util.spec_from_file_location("clef_upstream_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, help="local pinned Clef snapshot")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--upstream", type=Path, help="unmodified joint_schema_model.py from this revision")
    parser.add_argument("--output", type=Path, default=HERE.parent / "tmp" / "golden-clef.json")
    parser.add_argument("--cases", nargs="+", default=["readme_ticket", "email_object_state"])
    parser.add_argument("--offload-dir", type=Path, help="CPU inference with disk-backed decoder layers")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--hidden-output", type=Path, help="optional F32 final hidden rows for debugging")
    args = parser.parse_args()
    if args.threads < 1 or args.max_tokens < 1:
        parser.error("--threads and --max-tokens must be positive")
    if args.offload_dir and args.device != "cpu":
        parser.error("--offload-dir uses CPU execution")

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    torch.set_num_threads(args.threads)
    wanted = parse_case_ids(args.cases)
    cases = [case for case in CASES if wanted is None or case["id"] in wanted]
    directory = args.local or Path(snapshot_download(MODEL, revision=args.revision))
    source = args.upstream or directory / "joint_schema_model.py"
    upstream, source_hash = source_module(source, args.revision)
    tokenizer = transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True)
    config = transformers.AutoConfig.from_pretrained(directory, local_files_only=True)
    kwargs = dict(dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager")
    if args.offload_dir:
        args.offload_dir.mkdir(parents=True, exist_ok=True)
        prefix = "model.language_model"
        kwargs.update(
            device_map={
                f"{prefix}.embed_tokens": "cpu",
                f"{prefix}.norm": "cpu",
                f"{prefix}.rotary_emb": "cpu",
                "lm_head": "cpu",
                "model.visual": "disk",
                **{f"{prefix}.layers.{i}": "disk" for i in range(config.text_config.num_hidden_layers)},
            },
            offload_folder=str(args.offload_dir),
            offload_state_dict=True,
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("--device cuda requires torch.cuda.is_available()")
        kwargs["device_map"] = {"": args.device}
    started = time.monotonic()
    print(f"loading {MODEL}@{args.revision}, BF16, {'CPU disk offload' if args.offload_dir else args.device}", flush=True)
    backbone = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(directory, **kwargs)
    backbone.config.use_cache = False
    head_config = json.loads((directory / "joint_head_config.json").read_text())
    head = upstream.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(directory / "joint_head.safetensors", device="cpu"), strict=True)
    head = head.to(device=args.device, dtype=torch.bfloat16)
    model = upstream.ClefModel(backbone, head).eval()
    print(f"loaded in {time.monotonic() - started:.1f}s", flush=True)
    output = {
        "source": {
            "repo": MODEL, "revision": args.revision,
            "upstream_sha256": source_hash,
            "dtype": "bfloat16", "device": args.device,
            "execution": "CPU decoder disk offload" if args.offload_dir else args.device,
            "quantization": None, "torch": torch.__version__, "transformers": transformers.__version__,
            "attention": "eager", "head": "unmodified upstream JointSchemaHead",
        },
        "cases": [],
    }
    for case in cases:
        record = upstream.encode_record(tokenizer, case, max_length=args.max_tokens)
        batch = upstream.collate_records([record], tokenizer.pad_token_id or 0, torch.device(args.device))
        started = time.monotonic()
        captured = {}
        hook = None
        if args.hidden_output:
            hook = backbone.model.language_model.register_forward_hook(
                lambda _module, _inputs, result: captured.update(hidden=result.last_hidden_state.detach().float().cpu())
            )
        try:
            with torch.inference_mode():
                raw_logits = model(batch)[0]
                logits = [values.float().cpu().tolist() for values in raw_logits]
                probabilities = [values.float().softmax(-1).cpu().tolist() for values in raw_logits]
        finally:
            if hook is not None:
                hook.remove()
        elapsed = time.monotonic() - started
        answers = {
            question.question_id: upstream.systemone_answer(
                case["questions"][question.question_id], dict(zip(question.option_ids, probs))
            )
            for question, probs in zip(record.questions, probabilities)
        }
        output["cases"].append({
            "id": case["id"], "state": case["state"], "questions": case["questions"],
            "input_ids": list(record.input_ids),
            "encoded_questions": [
                dict(id=q.question_id, kind=q.question_type, span=list(q.question_span), option_spans=[list(s) for s in q.option_spans], option_ids=list(q.option_ids))
                for q in record.questions
            ],
            "logits": logits, "probs": probabilities,
            "response": {"model": MODEL, "answers": answers, "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0}},
            "seconds": elapsed,
        })
        if args.hidden_output:
            args.hidden_output.mkdir(parents=True, exist_ok=True)
            captured["hidden"].numpy().astype("<f4").tofile(args.hidden_output / f"{case['id']}.f32")
        print(f"{case['id']}: {len(record.input_ids)} tokens, {elapsed:.1f}s, {json.dumps(answers)}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
