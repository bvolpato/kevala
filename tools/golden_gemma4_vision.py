#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch>=2.6", "transformers==5.19.0", "safetensors", "numpy"]
# [tool.uv]
# exclude-newer = "P7D"
# ///
"""Generate a bounded vision-only reference from pinned Transformers source.

Without --local, generate the tiny deterministic BF16-weight/F32-compute fixture
used by dev/gemma4-vision-check.html. With --local, load only an extracted vision
checkpoint such as tools/fetch_gemma_vision.py produces, never the decoder.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

SOURCE_COMMIT = "2c4914fb939fe9de0d8e7a798af4684d552f18b4"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
ROOT = Path(__file__).resolve().parent.parent


def pinned_module(kind: str):
    filename = f"{kind}_gemma4.py"
    path = Path(tempfile.gettempdir()) / f"kevala-{SOURCE_COMMIT}-{filename}"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/huggingface/transformers/{SOURCE_COMMIT}/src/transformers/models/gemma4/{filename}"
        with urlopen(url, timeout=30) as response:
            source = response.read(1024 * 1024 + 1)
        if len(source) > 1024 * 1024:
            raise ValueError("reference source exceeds its size limit")
        path.write_bytes(source)
    name = f"transformers.models.gemma4.{kind}_gemma4"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def source_models():
    # Import the package before replacing its modules, so relative imports resolve.
    import transformers.models.gemma4

    configuration, cfg_hash = pinned_module("configuration")
    modeling, model_hash = pinned_module("modeling")
    return configuration, modeling, {"configuration": cfg_hash, "modeling": model_hash}


def input_case(torch, patch: int, cols: int, rows: int, padded: int, reverse=False, name="rectangular", salt=0):
    positions = [(x, y) for y in range(rows) for x in range(cols)]
    positions += [(-1, -1)] * (padded - len(positions))
    pixels = torch.arange(padded * 3 * patch * patch, dtype=torch.float32)
    pixels = ((pixels * 17 + 31 + salt) % 251 / 250).reshape(padded, -1)
    positions = torch.tensor(positions, dtype=torch.int64)
    if reverse:
        positions = positions.flip(0)
        pixels = pixels.flip(0)
    return {"id": "reordered" if reverse else name, "pixels": pixels, "positions": positions, "softTokens": rows * cols // 9}


def encoded(tensor, dtype):
    import torch

    stored = tensor.detach().contiguous().to(torch.bfloat16 if dtype == "bf16" else torch.float32)
    raw = stored.view(torch.uint8).numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def evaluate(torch, vision, projector, cases):
    output = []
    with torch.inference_mode():
        for case in cases:
            hidden = vision(pixel_values=case["pixels"][None], pixel_position_ids=case["positions"][None]).last_hidden_state
            projected = projector(hidden)
            if projected.shape != (case["softTokens"], projector.text_hidden_size) or not torch.isfinite(projected).all():
                raise ValueError(f"invalid projected rows for {case['id']}: {projected.shape}")
            output.append({
                "id": case["id"], "softTokens": case["softTokens"],
                "patches": encoded(case["pixels"], "f32"),
                "positions": base64.b64encode(case["positions"].to(torch.int32).numpy().tobytes()).decode("ascii"),
                "projected": encoded(projected, "f32"),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, help="directory containing vision-only model.safetensors and config.json")
    parser.add_argument("--camera", action="store_true", help="generate 320x240 camera-grid cases; requires --local")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/gemma4-vision-tiny.json")
    args = parser.parse_args()
    if args.camera and not args.local:
        parser.error("--camera requires --local")
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(4914)
    configuration, modeling, hashes = source_models()
    checkpoint = None
    if args.local:
        manifest = json.loads((args.local / "manifest.json").read_text())
        if manifest["source"]["repo"] != "google/gemma-4-E2B-it" or manifest["source"]["revision"] != MODEL_REVISION:
            raise ValueError("vision extraction does not match the pinned E2B checkpoint")
        checkpoint = manifest["files"]["model.safetensors"]
        with (args.local / "model.safetensors").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != checkpoint["sha256"] or (args.local / "model.safetensors").stat().st_size != checkpoint["bytes"]:
            raise ValueError("vision checkpoint differs from the extraction manifest")
        config = json.loads((args.local / "config.json").read_text())
        c = configuration.Gemma4VisionConfig(**config["vision_config"])
        text = configuration.Gemma4TextConfig(**config["text_config"])
        # Meta construction avoids allocating a second copy of the vision tower.
        with torch.device("meta"):
            vision = modeling.Gemma4VisionModel(c)
            projector = modeling.Gemma4MultimodalEmbedder(c, text)
        from safetensors import safe_open

        with safe_open(args.local / "model.safetensors", framework="pt", device="cpu") as source:
            keys = list(source.keys())
            if any(not key.startswith(("model.vision_tower.", "model.embed_vision.")) for key in keys):
                raise ValueError("--local must contain only extracted vision/projector tensors")
            for prefix, module in [("model.vision_tower.", vision), ("model.embed_vision.", projector)]:
                weights = {key[len(prefix):]: source.get_tensor(key).float() for key in keys if key.startswith(prefix)}
                module.load_state_dict(weights, strict=True, assign=True)
                del weights
        vision.encoder.rotary_emb = modeling.Gemma4VisionRotaryEmbedding(c)
        if args.camera:
            cases = [
                input_case(torch, c.patch_size, 12, 9, 144, name="camera-fast"),
                input_case(torch, c.patch_size, 21, 15, 360, name="camera-balanced"),
                input_case(torch, c.patch_size, 27, 21, 630, name="camera-detail"),
                input_case(torch, c.patch_size, 12, 9, 144, name="camera-fast-repeat", salt=1),
            ]
        else:
            cases = [input_case(torch, c.patch_size, 24, 24, 630)]
        tensors = None
    else:
        c = configuration.Gemma4VisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=32, patch_size=8,
            pooling_kernel_size=3, position_embedding_size=12, standardize=True,
            use_clipped_linears=True, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
            rope_parameters={"rope_type": "axial", "rope_theta": 100})
        text = configuration.Gemma4TextConfig(hidden_size=96)
        vision = modeling.Gemma4VisionModel(c)
        projector = modeling.Gemma4MultimodalEmbedder(c, text)
        with torch.no_grad():
            for prefix, module in [("vision_tower.", vision), ("embed_vision.", projector)]:
                for i, (name, value) in enumerate(module.state_dict().items()):
                    if name.endswith("_min"):
                        data = torch.full_like(value, -0.45 - (i % 3) * 0.1)
                    elif name.endswith("_max"):
                        data = torch.full_like(value, 0.35 + (i % 3) * 0.1)
                    elif name.endswith("std_bias"):
                        data = torch.linspace(-0.2, 0.3, value.numel()).reshape(value.shape)
                    elif name.endswith("std_scale") or value.ndim == 1:
                        data = torch.linspace(0.7, 1.3, value.numel()).reshape(value.shape)
                    else:
                        data = torch.randn_like(value) * (0.08 if "position_embedding" in name else 0.12)
                    value.copy_(data.bfloat16().float())
        tensors = []
        for prefix, module in [("vision_tower.", vision), ("embed_vision.", projector)]:
            for name, value in module.state_dict().items():
                dtype = "bf16" if value.ndim >= 2 else "f32"
                tensors.append({"name": prefix + name.removesuffix(".weight"), "shape": list(value.shape) or [1], "dtype": dtype, "data": encoded(value.reshape(value.shape or [1]), dtype)})
        cases = [input_case(torch, c.patch_size, 6, 3, 36), input_case(torch, c.patch_size, 6, 3, 36, True),
            input_case(torch, c.patch_size, 3, 3, 18, name="resized-square")]
    c._attn_implementation = "eager"
    vision.eval()
    projector.eval()
    fixture = {"sourceCommit": SOURCE_COMMIT, "sourceHashes": hashes, "compute": "f32",
        "model": "google/gemma-4-E2B-it" if args.local else "tiny-deterministic", "revision": MODEL_REVISION if args.local else None,
        "config": {"arch": "gemma4-vision", "vision_config": c.to_dict(), "text_hidden_size": text.hidden_size},
        "cases": evaluate(torch, vision, projector, cases)}
    if tensors is not None:
        fixture["tensors"] = tensors
    if checkpoint is not None:
        fixture["checkpoint"] = checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {args.output}: {len(fixture['cases'])} cases; source {SOURCE_COMMIT}; BF16 stored weights, F32 compute")


if __name__ == "__main__":
    main()
