#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
"""Generate a small parity fixture from the unmodified upstream Clef head.

uv run tools/golden_clef_head.py /path/to/joint_schema_model.py -o tests/fixtures/clef-head.json
Only a tiny synthetic head is instantiated. No released backbone weights are loaded.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("clef_reference", args.upstream)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.manual_seed(731)
    config = dict(hidden_size=8, width=8, routing_layers=2, layers=2, heads=2, feedforward=16)
    head = module.JointSchemaHead(**config).eval()
    with torch.no_grad():
        head.prior_logit_scale.fill_(0.3)
        head.joint_logit_scale.fill_(-0.2)
        head.residual_gate.fill_(0.7)
    questions = (
        module.EncodedQuestion("yes", 0, (1, 3), ((3, 5), (5, 6)), ("true", "false")),
        module.EncodedQuestion("choice", 1, (6, 8), ((8, 9), (9, 10), (10, 12)), ("a", "b", "c")),
        module.EncodedQuestion("score", 2, (2, 4), ((4, 5), (5, 7)), ("0", "1")),
    )
    ids = tuple([2, 6, 4, 1, 0, 9, 8, 3, 5, 7, 10, 11, 12])
    record = module.EncodedRecord(ids, questions, "tiny")
    hidden = torch.randn(1, len(ids), config["hidden_size"])
    lexical = torch.randn(16, config["hidden_size"])
    batch = module.collate_records([record], 0, torch.device("cpu"))
    with torch.inference_mode():
        logits = head(hidden, batch["input_ids"], batch["attention_mask"], [record], lexical)[0]

    class ByteTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return SimpleNamespace(input_ids=list(text.encode("utf-8")))

    request = {
        "state": {"z": [True, 1.0], "a": "café"},
        "questions": {
            "route": {"type": "choice", "instructions": {"z": 2, "a": "where"}, "criteria": {"z": None, "a": "first"}},
            "yes": {"type": "noul", "instructions": "", "criteria": {"false": "Not yet"}},
            "level": {"type": "score", "criteria": ["low", "high"]},
        },
    }
    encoded = module.encode_record(ByteTokenizer(), request, max_length=4096)
    encoded_truncated = module.encode_record(ByteTokenizer(), request, max_length=4096, max_state_tokens=5)
    fixture = {
        "upstream": "Cloudflare/clef-flash",
        "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        "config": config,
        "tensors": {name: {"shape": list(value.shape), "values": value.flatten().tolist()} for name, value in head.state_dict().items()},
        "input_ids": list(ids),
        "questions": [dict(id=q.question_id, kind=q.question_type, span=list(q.question_span), option_spans=[list(x) for x in q.option_spans], option_ids=list(q.option_ids)) for q in questions],
        "hidden": hidden.flatten().tolist(),
        "lexical": lexical.flatten().tolist(),
        "lexical_shape": list(lexical.shape),
        "logits": [x.tolist() for x in logits],
        "encoding_request": request,
        "encoding": {"ids": list(encoded.input_ids), "questions": [dict(span=list(q.question_span), option_spans=[list(x) for x in q.option_spans], option_ids=list(q.option_ids)) for q in encoded.questions]},
        "truncated_ids": list(encoded_truncated.input_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {args.output}: {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
