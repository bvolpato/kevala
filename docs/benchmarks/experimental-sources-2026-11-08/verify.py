#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import argparse
import ast
import hashlib
import json
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path


MAX_OBJECT_BYTES = 2 * 1024 * 1024
PROCESS_TIMEOUT_SECONDS = 15
LANGUAGES = {"javascript", "python", "json", "toml", "rust", "text"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def is_nonnegative_int(value):
    return type(value) is int and value >= 0


def valid_sha256(value):
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def diagnostic(output):
    excerpt = output.decode("utf-8", "replace").strip()
    return excerpt[:1200] or "no diagnostic output"


def run_process(command, *, input_bytes=None, cwd=None):
    try:
        result = subprocess.run(
            command,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PROCESS_TIMEOUT_SECONDS,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError:
        return None, f"command not found: {command[0]}"
    except subprocess.TimeoutExpired:
        return None, f"command timed out after {PROCESS_TIMEOUT_SECONDS} seconds: {command[0]}"

    if result.returncode != 0:
        output = result.stderr or result.stdout
        return None, f"{command[0]} exited with status {result.returncode}: {diagnostic(output)}"
    return result, None


def verify_archive(syntax_requested):
    root = Path(__file__).absolute().parent
    errors = []
    syntax_checks = []
    source_file_count = 0
    manifest = None
    object_records = {}
    object_data = {}
    references = []

    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        errors.append(f"archive directory is unavailable: {exc}")
        root_mode = 0
    if root_mode and (stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode)):
        errors.append("archive path must be a regular directory, not a symlink")

    manifest_path = root / "manifest.json"
    try:
        manifest_mode = manifest_path.lstat().st_mode
    except OSError as exc:
        errors.append(f"manifest.json is unavailable: {exc}")
        manifest_mode = 0
    if manifest_mode and (stat.S_ISLNK(manifest_mode) or not stat.S_ISREG(manifest_mode)):
        errors.append("manifest.json must be a regular file, not a symlink")
    elif manifest_mode:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"manifest.json cannot be read as UTF-8 JSON: {exc}")

    objects_dir = root / "objects"
    try:
        objects_mode = objects_dir.lstat().st_mode
    except OSError as exc:
        errors.append(f"objects directory is unavailable: {exc}")
        objects_mode = 0
    if objects_mode and (stat.S_ISLNK(objects_mode) or not stat.S_ISDIR(objects_mode)):
        errors.append("objects must be a regular directory, not a symlink")

    if not isinstance(manifest, dict):
        errors.append("manifest root must be a JSON object")
    else:
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            errors.append("manifest schema_version must be 1")
        if not isinstance(manifest.get("base_commit"), str):
            errors.append("manifest base_commit must be a string")

        experiments = manifest.get("experiments")
        if not isinstance(experiments, list):
            errors.append("manifest experiments must be an array")
            experiments = []
        for exp_index, experiment in enumerate(experiments):
            label = f"experiments[{exp_index}]"
            if not isinstance(experiment, dict):
                errors.append(f"{label} must be an object")
                continue
            if not isinstance(experiment.get("id"), str):
                errors.append(f"{label}.id must be a string")
            if not isinstance(experiment.get("base_commit"), str):
                errors.append(f"{label}.base_commit must be a string")
            if "validation" not in experiment:
                errors.append(f"{label}.validation is missing")
            files = experiment.get("files")
            if not isinstance(files, list):
                errors.append(f"{label}.files must be an array")
                continue
            for file_index, source in enumerate(files):
                source_label = f"{label}.files[{file_index}]"
                if not isinstance(source, dict):
                    errors.append(f"{source_label} must be an object")
                    continue
                source_file_count += 1
                if not isinstance(source.get("path"), str) or not source["path"]:
                    errors.append(f"{source_label}.path must be a non-empty string")
                if not is_nonnegative_int(source.get("bytes")):
                    errors.append(f"{source_label}.bytes must be a non-negative integer")
                elif source["bytes"] > MAX_OBJECT_BYTES:
                    errors.append(f"{source_label}.bytes exceeds the 2 MiB object limit")
                if not valid_sha256(source.get("sha256")):
                    errors.append(f"{source_label}.sha256 must be a lowercase SHA-256 digest")
                references.append((source_label, source))

        objects = manifest.get("objects")
        if not isinstance(objects, list):
            errors.append("manifest objects must be an array")
            objects = []
        for object_index, record in enumerate(objects):
            label = f"objects[{object_index}]"
            if not isinstance(record, dict):
                errors.append(f"{label} must be an object")
                continue
            digest = record.get("sha256")
            path = record.get("path")
            if not valid_sha256(digest):
                errors.append(f"{label}.sha256 must be a lowercase SHA-256 digest")
                continue
            expected_path = f"objects/{digest}.txt"
            if path != expected_path:
                errors.append(f"{label}.path must be exactly {expected_path}")
            if expected_path in object_records:
                errors.append(f"{label}.path duplicates an earlier object record: {expected_path}")
            else:
                object_records[expected_path] = record
            if not is_nonnegative_int(record.get("bytes")):
                errors.append(f"{label}.bytes must be a non-negative integer")
            elif record["bytes"] > MAX_OBJECT_BYTES:
                errors.append(f"{label}.bytes exceeds the 2 MiB object limit")
            languages = record.get("languages")
            if (
                not isinstance(languages, list)
                or not languages
                or any(not isinstance(language, str) or language not in LANGUAGES for language in languages)
                or len(set(languages)) != len(languages)
            ):
                errors.append(f"{label}.languages must be a non-empty list of unique supported languages")

        used_objects = set()
        for source_label, source in references:
            digest = source.get("sha256")
            object_path = source.get("object_path")
            expected_path = f"objects/{digest}.txt" if valid_sha256(digest) else None
            if object_path != expected_path:
                errors.append(f"{source_label}.object_path must be exactly {expected_path}")
            if not isinstance(object_path, str) or object_path not in object_records:
                errors.append(f"{source_label} references an object absent from manifest.objects")
                continue
            used_objects.add(object_path)
            record = object_records[object_path]
            if source.get("sha256") != record.get("sha256"):
                errors.append(f"{source_label}.sha256 does not match its object record")
            if source.get("bytes") != record.get("bytes"):
                errors.append(f"{source_label}.bytes does not match its object record")
        for path in object_records:
            if path not in used_objects:
                errors.append(f"manifest object is not referenced by a source file: {path}")

    expected_paths = set(object_records)
    if objects_mode and stat.S_ISDIR(objects_mode) and not stat.S_ISLNK(objects_mode):
        try:
            for entry in objects_dir.iterdir():
                entry_path = f"objects/{entry.name}"
                try:
                    entry_mode = entry.lstat().st_mode
                except OSError as exc:
                    errors.append(f"cannot inspect {entry_path}: {exc}")
                    continue
                if stat.S_ISLNK(entry_mode):
                    errors.append(f"object path must not be a symlink: {entry_path}")
                elif not stat.S_ISREG(entry_mode):
                    errors.append(f"object path must be a regular file: {entry_path}")
                elif entry_path not in expected_paths:
                    errors.append(f"unlisted file in objects directory: {entry_path}")
        except OSError as exc:
            errors.append(f"cannot list objects directory: {exc}")

    for object_path, record in object_records.items():
        object_file = root / object_path
        try:
            mode = object_file.lstat().st_mode
        except OSError as exc:
            errors.append(f"{object_path} is unavailable: {exc}")
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            errors.append(f"{object_path} must be a regular file, not a symlink")
            continue
        actual_size = object_file.stat().st_size
        if actual_size > MAX_OBJECT_BYTES:
            errors.append(f"{object_path} exceeds the 2 MiB object limit")
            continue
        try:
            data = object_file.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read {object_path}: {exc}")
            continue
        actual_hash = hashlib.sha256(data).hexdigest()
        if record.get("bytes") != len(data):
            errors.append(f"{object_path} size does not match manifest: expected {record.get('bytes')}, got {len(data)}")
        if record.get("sha256") != actual_hash:
            errors.append(f"{object_path} SHA-256 does not match manifest: got {actual_hash}")
        object_data[object_path] = data

    if syntax_requested and not errors:
        rustfmt_path = None
        if any("rust" in record["languages"] for record in object_records.values()):
            rustup_result, rustup_error = run_process(
                ["rustup", "which", "--toolchain", "1.95.0", "rustfmt"], cwd=root
            )
            if rustup_error:
                errors.append(f"cannot locate rustfmt: {rustup_error}")
            else:
                rustfmt_path = rustup_result.stdout.decode("utf-8", "replace").strip()
                if not rustfmt_path:
                    errors.append("rustup which returned an empty rustfmt path")

        for object_path, record in object_records.items():
            data = object_data[object_path]
            try:
                source_text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{object_path} is not valid UTF-8: {exc}")
                continue
            for language in record["languages"]:
                try:
                    if language == "python":
                        ast.parse(source_text, filename=object_path)
                    elif language == "json":
                        json.loads(source_text)
                    elif language == "toml":
                        tomllib.loads(source_text)
                    elif language == "javascript":
                        _, error = run_process(
                            ["node", "--check", "--input-type=module", "-"],
                            input_bytes=data,
                            cwd=root,
                        )
                        if error:
                            raise ValueError(error)
                    elif language == "rust" and rustfmt_path:
                        _, error = run_process(
                            [
                                rustfmt_path,
                                "--edition",
                                "2021",
                                "--emit",
                                "stdout",
                                "--config",
                                "skip_children=true",
                            ],
                            input_bytes=data,
                            cwd=root,
                        )
                        if error:
                            raise ValueError(error)
                    elif language == "text":
                        pass
                except (SyntaxError, ValueError, RecursionError) as exc:
                    errors.append(f"syntax check failed for {object_path} ({language}): {exc}")
                else:
                    if language != "rust" or rustfmt_path:
                        syntax_checks.append({"object_path": object_path, "language": language})

    summary = {
        "ok": not errors,
        "archive": str(root),
        "base_commit": manifest.get("base_commit") if isinstance(manifest, dict) else None,
        "experiments": len(manifest.get("experiments", [])) if isinstance(manifest, dict) and isinstance(manifest.get("experiments"), list) else 0,
        "source_files": source_file_count,
        "objects": len(object_records),
        "syntax_requested": syntax_requested,
        "syntax_checks": syntax_checks,
        "errors": errors,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Verify the inert experimental source archive.")
    parser.add_argument("--syntax", action="store_true", help="run syntax-only parsers for supported languages")
    args = parser.parse_args()
    summary = verify_archive(args.syntax)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
