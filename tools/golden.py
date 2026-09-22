#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.6", "transformers>=5.0", "safetensors", "numpy", "tokenizers>=0.21"]
# ///
"""Dev-time reference dump. Never shipped, never needed at runtime.

Runs the original PyTorch Laya checkpoint through the upstream `laya` SDK code and writes what
the Rust engine is tested against:

  tests/fixtures/golden.json     token ids, marker positions, raw logits, probabilities, act
                                 probabilities and the SDK's full response for every fixture
  tests/fixtures/tokenizer.json  token ids from the Hugging Face `tokenizers` library for a
                                 corpus of awkward strings
  tmp/golden/*.f32               per-layer hidden states for one sequence (debugging only)

Usage: uv run tools/golden.py --model tmp/laya --sdk tmp/laya-sdk
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LOREM = ("The customer wrote in about a recurring charge they did not recognise. "
         "They have contacted support three times already and are threatening a chargeback. ")

# Cases beyond the laya-web parity set: each one pins a branch the port must reproduce.
EXTRA_CASES = [
    {"id": "readme_email_multi", "state": {
        "from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
     "questions": {
        "department": {"type": "choice", "instructions": "Which department should handle this request?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                    "sales": "pricing, new contracts", "other": "everything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is this request?",
                    "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
        "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}}},
    # python json.dumps number formatting: floats keep repr, ints stay ints
    {"id": "state_numbers", "state": {"score": 0.5, "big": 1e20, "small": 1e-07, "int": 3, "neg": -2.5,
                                      "exact": 1.0, "tiny": 5e-324, "ok": True, "none": None, "pi": 3.141592653589793,
                                      "sci": 12345678901234567890.0, "frac": 0.0001, "frac2": 0.00001},
     "questions": {"q": {"type": "noul", "instructions": "Is any value negative?"}}},
    # conversation list as the state
    {"id": "state_list", "state": [{"role": "user", "content": "my order is late"},
                                   {"role": "agent", "content": "sorry, checking now"},
                                   {"role": "user", "content": "cancel it, I'm done"}],
     "questions": {"q": {"type": "noul", "instructions": "Does the user want to cancel?"}}},
    # laya 0.3.5: structured criterion values become compact JSON; 0 and False are real descriptions
    {"id": "criteria_structured", "state": "The pull request adds a feature flag and tests.",
     "questions": {"q": {"type": "choice", "instructions": "How risky is this change?",
                         "criteria": {"low": {"desc": "tests and a flag"}, "medium": 0, "high": False, "unknown": ""}}}},
    {"id": "score_structured", "state": "Revenue grew 3% and churn fell.",
     "questions": {"q": {"type": "score", "instructions": "How good is the quarter?",
                         "criteria": ["bad", {"label": "flat"}, "good"]}}},
    # instructions as a non-string: json.dumps with ensure_ascii=True
    {"id": "instructions_object", "state": "café crème — très bien",
     "questions": {"q": {"type": "noul", "instructions": {"ask": "Is the review positive?", "lang": "français"}}}},
    # single-option choice (laya 0.3.5 pads top2 with zero)
    {"id": "choice_k1", "state": "Hello there",
     "questions": {"q": {"type": "choice", "instructions": "Pick one.", "criteria": {"only": "the only option"}}}},
    # ~200 tokens: longer than the 128-token sliding window, shorter than the budget
    {"id": "state_sliding_window", "state": LOREM * 6,
     "questions": {"q": {"type": "choice", "instructions": "Route this ticket.",
                         "criteria": {"billing": "payments", "technical": "bugs", "retention": "cancellations"}},
                   "r": {"type": "noul", "instructions": "Is the customer threatening a chargeback?"}}},
    # NFC normalisation: decomposed e + combining acute must match the precomposed form
    {"id": "nfc_decomposed", "state": "Café au lait, résumé attached",
     "questions": {"q": {"type": "noul", "instructions": "Is a document attached?"}}},
    # special tokens typed into the state are real special tokens to the tokenizer
    {"id": "special_tokens_in_state", "state": "hello [SEP] world [CLS] <|endoftext|> |||EMAIL_ADDRESS|||",
     "questions": {"q": {"type": "noul", "instructions": "Does this contain an email address?"}}},
    # a game-style perception question, the shape the demos use
    {"id": "game_perception", "state": "The piece would land flat on the left side, leaving no holes and clearing one line.",
     "questions": {"q": {"type": "noul", "instructions": "Does the stack look clean after this move?",
                         "criteria": {"true": "clean: no holes, flat surface", "false": "messy: holes or tall towers"}}}},
]

TOKENIZER_CORPUS = [
    "", " ", "  ", "   ", "\n", "\n\n", "\t", " \t\n ", "a", " a", "a ", "hello world", "Hello, World!",
    "don't won't I'm you're they've we'll he'd IT'S", "'s 's 'S", "numbers 123 4567 89012 3.14159 1,000,000",
    "line one\n\n\tline two   with    gaps\r\nline three\n", "trailing spaces    ", "    leading spaces",
    "a  b   c    d     e      f", " " * 30 + "x", "x" + " " * 30, "tab\t\tstop", "mixed  nbsp em space",
    "café naïve résumé coöperate", "Café decomposed", "Å angstrom sign", "ＡＢ fullwidth",
    "日本語のテキスト 完全に壊れている", "中文字符测试", "한국어 텍스트", "مرحبا بالعالم", "שלום עולם",
    "Ελληνικά κείμενο", "Русский текст", "हिन्दी पाठ", "ខ្ញុំចង់បិទគណនីរបស់ខ្ញុំ", "ภาษาไทย",
    "🔥🔥 this is broken 🔥", "👨‍👩‍👧‍👦 family emoji", "🇧🇷 flag", "math ∑∫√∞ ≤ ≥ ≠", "symbols @#$%^&*()_+-=[]{}|;:'\",.<>/?`~",
    "[CLS] [SEP] [MASK] [PAD] [UNK] <|endoftext|> <|padding|>", "x[SEP]y", "a [MASK] b  [MASK]c", "|||IP_ADDRESS||| |||PHONE_NUMBER|||",
    "[unused0] [unused82]", "email me at someone@example.com or call +1 (555) 010-9999",
    "https://example.com/path?query=1&x=y#frag", "snake_case camelCase PascalCase kebab-case SCREAMING_CASE",
    "{\"key\": \"value\", \"n\": 1.5, \"list\": [1, 2, 3]}", "    def f(x):\n        return x ** 2\n",
    "¡¿Qué?! «guillemets» „quotes“ ‘single’", "ﬁ ligature ﬀ", "ℌ𝔢𝔩𝔩𝔬 math letters 𝟙𝟚𝟛", "​zero‌width‍",
    "ẹ́̂ stacked marks", "각 hangul jamo", "Ǆ ǅ ǆ digraphs", "Ⅻ roman numeral ½ ¾ ²",
    "٣٤٥ arabic digits ١٢", "a\u0000b control", "end.", "...", "!!!???", "--", "—", "a—b–c",
    LOREM * 3,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "tmp", "laya"))
    ap.add_argument("--sdk", default=os.path.join(ROOT, "tmp", "laya-sdk"))
    ap.add_argument("--fixtures", default=os.path.join(ROOT, "tmp", "fixtures-laya-web.json"))
    ap.add_argument("--dump-case", default="state_sliding_window")
    a = ap.parse_args()

    sys.path.insert(0, a.sdk)
    import laya
    from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket

    torch.manual_seed(0)
    agent = laya.load(a.model, device="cpu")
    tok, model = agent.tok, agent.model
    max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)

    cases = json.load(open(a.fixtures)) + EXTRA_CASES
    out = {"source": {"model": "convaiinnovations/laya", "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
                      "sdk": "laya " + laya.__version__, "sdk_commit": "573e5b62696ba441230cd6be71d593331b5d23af",
                      "precision": "float32 compute on the float16 checkpoint", "torch": torch.__version__},
           "temperature": agent.temperature, "temperature_by_options": agent.temperature_by_options,
           "cases": []}

    for case in cases:
        rec = {"id": case["id"], "state": case["state"], "questions": case["questions"], "expect": {}}
        items = []
        for qid, qdef in case["questions"].items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(tok, case["state"], q, max_len, head_max_len)
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            rec["expect"][qid] = {"input_ids": ids, "marker_pos": markers, "options": render_options(q)}
        # one sequence at a time: the reference for each question in isolation
        for (qid, e), it in zip(rec["expect"].items(), items):
            b = collate_items([[it]], tok.pad_token_id)
            with torch.no_grad():
                logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
            k = len(it["markers"])
            e["logits"] = [float(v) for v in logits[0, :k]]
            e["act_logits"] = [float(v) for v in act[0]]
        rec["response"] = agent.predict(case["state"], case["questions"])
        out["cases"].append(rec)
        print("%-28s %s" % (case["id"], "  ".join("%s L=%d" % (qid, len(e["input_ids"])) for qid, e in rec["expect"].items())))

    with open(os.path.join(ROOT, "tests", "fixtures", "golden.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)

    # tokenizer corpus straight from the HF tokenizers library
    tk = tok.backend_tokenizer if hasattr(tok, "backend_tokenizer") else tok._tokenizer
    corpus = TOKENIZER_CORPUS + [c["state"] for c in cases if isinstance(c["state"], str)]
    toks = [{"text": s, "ids": tk.encode(s, add_special_tokens=False).ids} for s in corpus]
    with open(os.path.join(ROOT, "tests", "fixtures", "tokenizer.json"), "w") as f:
        json.dump(toks, f, ensure_ascii=False)

    # per-layer activations for one case, for localising a mismatch in the Rust port
    dump = next(c for c in out["cases"] if c["id"] == a.dump_case)
    qid = next(iter(dump["expect"]))
    e = dump["expect"][qid]
    ddir = os.path.join(ROOT, "tmp", "golden")
    os.makedirs(ddir, exist_ok=True)
    ids = torch.tensor([e["input_ids"]])
    att = torch.ones_like(ids)
    enc = model.encoder
    acts = {}
    hooks = [enc.embeddings.register_forward_hook(lambda m, i, o: acts.__setitem__("emb", o[0].clone()))]
    for li, layer in enumerate(enc.layers):
        hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: acts.__setitem__("layer%02d" % li, (o[0] if isinstance(o, tuple) else o)[0].clone())))
    for li, layer in enumerate(model.head.layers):
        hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: acts.__setitem__("head%d" % li, o[0].clone())))
    with torch.no_grad():
        final = enc(input_ids=ids, attention_mask=att).last_hidden_state[0]
        acts["final"] = final.clone()
        mpos = torch.tensor([e["marker_pos"]])
        model(ids, att, mpos, torch.ones_like(mpos, dtype=torch.bool), torch.tensor([QTYPES[dump["questions"][qid]["type"]]]))
    for h in hooks:
        h.remove()
    meta = {"case": a.dump_case, "question": qid, "input_ids": e["input_ids"], "tensors": {}}
    for k, v in acts.items():
        arr = v.float().numpy().astype("<f4")
        arr.tofile(os.path.join(ddir, k + ".f32"))
        meta["tensors"][k] = list(arr.shape)
    json.dump(meta, open(os.path.join(ddir, "meta.json"), "w"))
    print("wrote golden.json (%d cases), tokenizer.json (%d strings), %d activation dumps" % (len(out["cases"]), len(toks), len(acts)))


if __name__ == "__main__":
    main()
