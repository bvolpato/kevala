#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub", "transformers>=5.17"]
# ///
"""Convert pinned optional models serially; publishing is a separate explicit flag.

Build the native CLI first: cargo build --release -p kevala-cli
Run: uv run tools/convert_models.py --models kev-4b semif-qwen3.5-0.8b
After validation: uv run tools/convert_models.py --models kev-4b --upload-only --publish bvolpato/kevala-packs
"""
import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "tools/model-sources.json").read_text())
PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.jinja", "LICENSE*", "merges.txt", "vocab.json"]


def snapshot(spec):
    return snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=PATTERNS, max_workers=1)


def inspect(path, source):
    with path.open("rb") as stream:
        prefix = stream.read(16)
        if prefix[:4] != b"KVLA" or struct.unpack_from("<I", prefix, 4)[0] != 1:
            raise ValueError(f"{path}: invalid pack")
        header = json.loads(stream.read(struct.unpack_from("<I", prefix, 8)[0]))
        stream.seek(0)
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if header["model"]["revision"] != source["revision"]:
        raise ValueError(f"{path}: source revision does not match tools/model-sources.json")
    if header["model"].get("base_revision") != source.get("base", source)["revision"]:
        raise ValueError(f"{path}: base revision does not match tools/model-sources.json")
    if "method" in source and header["model"].get("method_revision") != source["method"]["revision"]:
        raise ValueError(f"{path}: method revision does not match tools/model-sources.json")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha,
            "source": source, "model": header["model"], "config": header["config"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(SOURCES), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp")
    parser.add_argument("--cli", type=Path, default=ROOT / "target/release/kevala")
    parser.add_argument("--publish", metavar="HF_REPO", help="upload converted packs and per-pack manifests to this repository")
    parser.add_argument("--upload-only", action="store_true", help="publish existing validated packs without converting")
    parser.add_argument("--model-card", type=Path, help="include this README and repository license notices in the upload")
    args = parser.parse_args()
    if args.upload_only and not args.publish:
        parser.error("--upload-only requires --publish")
    if args.model_card and not args.publish:
        parser.error("--model-card requires --publish")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operations = []
    for name in args.models:
        spec = SOURCES[name]
        path = args.output_dir / f"{name}-q8.kevala"
        if not args.upload_only:
            source = snapshot(spec)
            base = snapshot(spec["base"]) if spec["kind"] == "kev" else source
            command = [str(args.cli), f"convert-{spec['kind']}", "--base", base, "--name", name,
                       "--source", spec["repo"], "--base-source", spec.get("base", spec)["repo"],
                       "--base-revision", spec.get("base", spec)["revision"], "-o", str(path)]
            if spec["kind"] == "kev":
                command += ["--kev", source, "--kev-revision", spec["revision"]]
                subprocess.run(command, check=True, cwd=ROOT)
            else:
                from transformers import AutoTokenizer

                # Qwen2Tokenizer changes the raw checkpoint's pretokenizer on load.
                with tempfile.TemporaryDirectory(prefix="kevala-tokenizer-") as tokenizer_dir:
                    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(tokenizer_dir)
                    command += ["--tokenizer", str(Path(tokenizer_dir) / "tokenizer.json"),
                                "--method-revision", spec["method"]["revision"]]
                    subprocess.run(command, check=True, cwd=ROOT)
        record = inspect(path, spec)
        manifest = path.with_suffix(".json")
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ("file", "bytes", "sha256")}), flush=True)
        operations += [CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path),
                       CommitOperationAdd(path_in_repo=manifest.name, path_or_fileobj=manifest)]
    if args.publish:
        if args.model_card:
            operations += [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=args.model_card),
                           CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=ROOT / "LICENSE"),
                           CommitOperationAdd(path_in_repo="THIRD_PARTY_NOTICES", path_or_fileobj=ROOT / "THIRD_PARTY_NOTICES")]
        api = HfApi()
        head = api.model_info(args.publish).sha
        result = api.create_commit(repo_id=args.publish, operations=operations, parent_commit=head,
                                   commit_message="Add pinned Kev and SemIf Qwen3.5 packs")
        print(f"Published {result.commit_url}", flush=True)


if __name__ == "__main__":
    main()
