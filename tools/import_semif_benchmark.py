#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Import the pinned SemIf decision fixture without downloading model weights.

The benchmark keeps the upstream JSONL bytes unchanged. This preserves SemIf's row IDs,
group IDs, labels, and provenance while making the source revision and hashes reviewable in
``benchmarks/decisions/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


SEMIF_REPOSITORY = "https://github.com/TheoLeeCJ/SemIf"
SEMIF_REVISION = "1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
SEMIF_LICENSE = "MIT"
SEMIF_LICENSE_SHA256 = "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be"
FILES = {
    "semif-authored144.jsonl": (
        "benchmarks/data/authored144.jsonl",
        "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079",
        144,
    ),
    "semif-perturbations108.jsonl": (
        "benchmarks/data/perturbations108.jsonl",
        "1dd7ccf80518d0e34886478ca23982aa726e9daccd343b9e95cedaf6b569bec4",
        108,
    ),
}
PERTURBATION_MANIFEST = {
    "path": "benchmarks/data/perturbations108-manifest.json",
    "sha256": "f81f93b57e0b46b0164b6eb6d65a419604c6af37a5aaba55c413f76581f2fef8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{path} is not a readable git checkout: {error}") from error


def read_rows(path: Path, expected: int, name: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"{name}: expected {expected} rows, found {len(rows)}")
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"{name}: every row needs a string id")
        if row["id"] in ids:
            raise ValueError(f"{name}: duplicate row id {row['id']}")
        ids.add(row["id"])
        options = row.get("options")
        label = row.get("label")
        if not isinstance(options, list) or len(options) != 3:
            raise ValueError(f"{name} row {row['id']}: expected exactly three options")
        option_ids = [option.get("id") for option in options]
        if any(not isinstance(option_id, str) or not option_id for option_id in option_ids):
            raise ValueError(f"{name} row {row['id']}: option ids must be nonempty strings")
        if len(set(option_ids)) != len(option_ids):
            raise ValueError(f"{name} row {row['id']}: duplicate option id")
        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(options):
            raise ValueError(f"{name} row {row['id']}: invalid label")
    return rows


def fixture_summary(path: Path, rows: list[dict]) -> dict:
    return {
        "path": path.name,
        "sha256": sha256(path),
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "options": dict(sorted(Counter(len(row["options"]) for row in rows).items())),
    }


def write_if_changed(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"refusing to replace existing file with different contents: {path}")
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="checked-out SemIf repository")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/decisions"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output_dir.resolve()
    actual_revision = git_revision(source)
    if actual_revision != SEMIF_REVISION:
        raise ValueError(f"SemIf revision mismatch: expected {SEMIF_REVISION}, found {actual_revision}")

    license_path = source / "LICENSE"
    if not license_path.is_file() or sha256(license_path) != SEMIF_LICENSE_SHA256:
        raise ValueError("SemIf LICENSE is missing or does not match the pinned MIT license")

    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for output_name, (source_name, expected_hash, expected_rows) in FILES.items():
        source_path = source / source_name
        if not source_path.is_file():
            raise ValueError(f"missing SemIf fixture: {source_path}")
        actual_hash = sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(f"{source_name} hash mismatch: expected {expected_hash}, found {actual_hash}")
        rows = read_rows(source_path, expected_rows, output_name)
        destination = output / output_name
        write_if_changed(destination, source_path.read_bytes())
        if sha256(destination) != expected_hash:
            raise ValueError(f"copied fixture hash mismatch: {destination}")
        summaries.append(fixture_summary(destination, rows))

    manifest_path = source / PERTURBATION_MANIFEST["path"]
    if sha256(manifest_path) != PERTURBATION_MANIFEST["sha256"]:
        raise ValueError("SemIf perturbation manifest hash mismatch")
    perturbation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if perturbation_manifest.get("source_sha256") != FILES["semif-authored144.jsonl"][1]:
        raise ValueError("SemIf perturbation manifest does not point at the pinned authored fixture")
    if perturbation_manifest.get("fixture_sha256") != FILES["semif-perturbations108.jsonl"][1]:
        raise ValueError("SemIf perturbation manifest does not point at the pinned perturbation fixture")

    kevala_path = output / "kevala-authored36.jsonl"
    kevala_summary = None
    if kevala_path.is_file():
        kevala_rows = read_rows(kevala_path, 36, "kevala-authored36.jsonl")
        family_counts = Counter(row.get("family") for row in kevala_rows)
        if family_counts != Counter({
            "candidate_selection": 12,
            "evidence_interpretation": 12,
            "rule_application": 12,
        }):
            raise ValueError(f"Kevala fixture family counts are not 12/12/12: {dict(family_counts)}")
        gold_positions = Counter(
            row["options"].index(next(option for option in row["options"] if option["id"] == row["gold_id"]))
            for row in kevala_rows
        )
        if gold_positions != Counter({0: 12, 1: 12, 2: 12}):
            raise ValueError(f"Kevala gold positions are not balanced: {dict(gold_positions)}")
        kevala_summary = fixture_summary(kevala_path, kevala_rows)

    manifest = {
        "version": "kevala-decision-v1",
        "permutations": ["identity", "rotate1", "rotate2"],
        "source_policy": "SemIf fixture bytes are copied unchanged after revision, license, and SHA-256 checks.",
        "semif": {
            "repository": SEMIF_REPOSITORY,
            "revision": SEMIF_REVISION,
            "license": SEMIF_LICENSE,
            "license_sha256": SEMIF_LICENSE_SHA256,
            "fixtures": summaries,
            "perturbation_manifest_sha256": PERTURBATION_MANIFEST["sha256"],
        },
        "kevala": kevala_summary,
        "counts": {
            "semif_authored": sum(item["rows"] for item in summaries if item["path"] == "semif-authored144.jsonl"),
            "semif_perturbations": sum(item["rows"] for item in summaries if item["path"] == "semif-perturbations108.jsonl"),
            "kevala_authored": kevala_summary["rows"] if kevala_summary else None,
        },
    }
    manifest["expanded_counts"] = {
        key: (value * len(manifest["permutations"]) if value is not None else None)
        for key, value in manifest["counts"].items()
    }
    manifest["expanded_counts"]["total"] = sum(
        value for key, value in manifest["expanded_counts"].items() if key != "total" and value is not None
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
