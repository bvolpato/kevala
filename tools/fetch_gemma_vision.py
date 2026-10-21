#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# [tool.uv]
# required-version = ">=0.13.4"
# exclude-newer = "P7D"
# ///
"""Fetch the pinned Gemma 4 E2B vision tower without downloading its decoder.

Run: uv run tools/fetch_gemma_vision.py --output-dir tmp/gemma4-vision/E2B
The extracted safetensors file preserves the original BF16 tensor bytes.
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from itertools import pairwise
from pathlib import Path

import requests

REPO = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
PREFIXES = ("model.vision_tower.", "model.embed_vision.")
EXPECTED_TENSORS = 659
EXPECTED_DATA_BYTES = 337_089_408
CHUNK_BYTES = 4 * 1024 * 1024
REQUEST_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024


class RemoteRanges:
    """Require exact, uncompressed ranges of one unchanged remote object."""

    def __init__(self, session, url):
        self.session = session
        self.url = url
        self.etag = None
        self.linked_etag = None
        self.total_bytes = None
        self.requests = 0

    def chunks(self, start, end):
        if not 0 <= start < end:
            raise ValueError("invalid source range")
        headers = {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}
        if self.etag is not None:
            headers["If-Match"] = self.etag
        with self.session.get(
            self.url, headers=headers, stream=True, timeout=(15, 120)
        ) as response:
            if response.status_code != 206:
                raise ValueError(
                    f"source range returned HTTP {response.status_code}, expected 206"
                )
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if match is None:
                raise ValueError("source range has an invalid Content-Range")
            first, last, total = map(int, match.groups())
            if (first, last) != (start, end - 1) or last >= total:
                raise ValueError(
                    "source Content-Range does not match the requested bytes"
                )
            etag = response.headers.get("ETag")
            if not etag or etag.startswith("W/"):
                raise ValueError("source range requires a strong ETag")
            linked = next(
                (
                    item.headers["X-Linked-ETag"]
                    for item in response.history
                    if "X-Linked-ETag" in item.headers
                ),
                None,
            )
            if self.etag is None:
                self.etag, self.linked_etag, self.total_bytes = etag, linked, total
            elif (etag, linked, total) != (
                self.etag,
                self.linked_etag,
                self.total_bytes,
            ):
                raise ValueError("source object identity changed between ranges")
            if (
                response.headers.get("Content-Encoding", "identity").lower()
                != "identity"
            ):
                raise ValueError("source range is compressed")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != end - start:
                raise ValueError(
                    "source Content-Length does not match the requested bytes"
                )
            self.requests += 1
            received = 0
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                received += len(chunk)
                if received > end - start or len(chunk) > CHUNK_BYTES:
                    raise ValueError("source range returned excess bytes")
                yield chunk
            if received != end - start:
                raise ValueError(
                    f"source range is truncated: received {received}, expected {end - start}"
                )

    def read(self, start, end):
        if end - start > MAX_JSON_BYTES:
            raise ValueError("metadata exceeds the bounded JSON size")
        return b"".join(self.chunks(start, end))


def fetch_json(session, filename):
    with session.get(
        f"{BASE_URL}/{filename}",
        headers={"Accept-Encoding": "identity"},
        stream=True,
        timeout=(15, 60),
    ) as response:
        if response.status_code != 200:
            raise ValueError(
                f"{filename} returned HTTP {response.status_code}, expected 200"
            )
        data = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if len(data) + len(chunk) > MAX_JSON_BYTES:
                raise ValueError(f"{filename} exceeds the bounded JSON size")
            data.extend(chunk)
    json.loads(data)
    return bytes(data)


def selected_tensors(header, data_bytes):
    selected = []
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        start, end = tensor["data_offsets"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= data_bytes
        ):
            raise ValueError(f"invalid source tensor offsets: {name}")
        intervals.append((start, end))
        if not name.startswith(PREFIXES):
            continue
        shape = tensor["shape"]
        if tensor["dtype"] != "BF16" or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(f"expected a BF16 vision tensor: {name}")
        if 2 * math.prod(shape) != end - start:
            raise ValueError(f"source tensor shape and byte count differ: {name}")
        selected.append((start, end, name, tensor))
    intervals.sort()
    if any(right[0] < left[1] for left, right in pairwise(intervals)):
        raise ValueError("source tensors overlap")
    selected.sort()
    if (
        len(selected) != EXPECTED_TENSORS
        or sum(end - start for start, end, _, _ in selected) != EXPECTED_DATA_BYTES
    ):
        raise ValueError("pinned vision tensor count or byte count does not match")
    return selected


def coalesced_ranges(selected):
    spans = []
    for start, end, _, _ in selected:
        if spans and spans[-1][1] == start:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    for start, end in spans:
        while start < end:
            stop = min(start + REQUEST_BYTES, end)
            yield start, stop
            start = stop


def output_header(source_header, selected):
    metadata = dict(source_header.get("__metadata__", {}))
    metadata.update(
        source_repo=REPO, source_revision=REVISION, selection="gemma4-vision-only"
    )
    output = {"__metadata__": metadata}
    offset = 0
    for start, end, name, tensor in selected:
        output[name] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + end - start],
        }
        offset += end - start
    encoded = json.dumps(output, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded, output


def temporary_file(directory, filename):
    return tempfile.NamedTemporaryFile(
        mode="w+b", dir=directory, prefix=f".{filename}-", suffix=".part", delete=False
    )


def install_file(part, target):
    # A hard link publishes atomically and refuses to replace another process's file.
    os.link(part, target)
    part.unlink()


def download(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "manifest.json",
    )
    if any((output_dir / name).exists() for name in filenames):
        raise ValueError(
            "output already contains an artifact; choose a new --output-dir"
        )
    session = requests.Session()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    parts = []
    try:
        config_bytes = fetch_json(session, "config.json")
        processor_bytes = fetch_json(session, "processor_config.json")
        config = json.loads(config_bytes)
        if (
            config["vision_config"]["hidden_size"] != 768
            or config["text_config"]["hidden_size"] != 1536
        ):
            raise ValueError(
                "pinned E2B vision and text widths differ from the extraction contract"
            )
        remote = RemoteRanges(session, f"{BASE_URL}/model.safetensors")
        header_bytes = struct.unpack("<Q", remote.read(0, 8))[0]
        if not 0 < header_bytes <= MAX_JSON_BYTES:
            raise ValueError("safetensors header exceeds the bounded JSON size")
        source_header_bytes = remote.read(8, 8 + header_bytes)
        header = json.loads(source_header_bytes)
        data_start = 8 + header_bytes
        selected = selected_tensors(header, remote.total_bytes - data_start)
        encoded_header, extracted_header = output_header(header, selected)
        print(
            f"Pinned source: {REPO}@{REVISION}; {len(selected)} BF16 tensors, "
            f"{EXPECTED_DATA_BYTES} data bytes",
            flush=True,
        )
        tensor_hashes = {name: hashlib.sha256() for _, _, name, _ in selected}
        artifact_hash = hashlib.sha256(encoded_header)
        data_hash = hashlib.sha256()
        completed = 0
        tensor_index = 0
        tensor_consumed = 0
        with temporary_file(output_dir, "model.safetensors") as stream:
            model_part = Path(stream.name)
            parts.append(model_part)
            stream.write(encoded_header)
            for start, end in coalesced_ranges(selected):
                for chunk in remote.chunks(data_start + start, data_start + end):
                    stream.write(chunk)
                    artifact_hash.update(chunk)
                    data_hash.update(chunk)
                    view = memoryview(chunk)
                    offset = 0
                    while offset < len(view):
                        first, last, name, _ = selected[tensor_index]
                        length = min(last - first - tensor_consumed, len(view) - offset)
                        tensor_hashes[name].update(view[offset : offset + length])
                        offset += length
                        tensor_consumed += length
                        if tensor_consumed == last - first:
                            tensor_index += 1
                            tensor_consumed = 0
                    completed += len(chunk)
                print(
                    f"Vision data: {completed}/{EXPECTED_DATA_BYTES} bytes", flush=True
                )
            if (
                completed != EXPECTED_DATA_BYTES
                or tensor_index != len(selected)
                or tensor_consumed
            ):
                raise ValueError("extracted vision data is incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        files = {
            "model.safetensors": {
                "bytes": model_part.stat().st_size,
                "sha256": artifact_hash.hexdigest(),
            },
            "config.json": {
                "bytes": len(config_bytes),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            "processor_config.json": {
                "bytes": len(processor_bytes),
                "sha256": hashlib.sha256(processor_bytes).hexdigest(),
            },
        }
        manifest = {
            "source": {
                "repo": REPO,
                "revision": REVISION,
                "file": "model.safetensors",
                "url": remote.url,
                "etag": remote.etag,
                "linked_etag": remote.linked_etag,
                "bytes": remote.total_bytes,
                "header_bytes": header_bytes,
                "header_sha256": hashlib.sha256(source_header_bytes).hexdigest(),
            },
            "extraction": {
                "prefixes": list(PREFIXES),
                "tensor_count": len(selected),
                "tensor_bytes": completed,
                "data_sha256": data_hash.hexdigest(),
                "range_requests": remote.requests,
                "max_stream_chunk_bytes": CHUNK_BYTES,
            },
            "files": files,
            "tensors": {
                name: {
                    **extracted_header[name],
                    "source_data_offsets": [start, end],
                    "bytes": end - start,
                    "sha256": tensor_hashes[name].hexdigest(),
                }
                for start, end, name, _ in selected
            },
        }
        staged = [(model_part, output_dir / "model.safetensors")]
        metadata = {
            "config.json": config_bytes,
            "processor_config.json": processor_bytes,
            "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
        }
        for filename, content in metadata.items():
            with temporary_file(output_dir, filename) as stream:
                part = Path(stream.name)
                parts.append(part)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((part, output_dir / filename))
        for part, target in staged:
            install_file(part, target)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    **files["model.safetensors"],
                    "tensor_count": len(selected),
                    "source_etag": remote.etag,
                }
            ),
            flush=True,
        )
        return manifest
    finally:
        session.close()
        for part in parts:
            part.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tmp/gemma4-vision/E2B",
    )
    args = parser.parse_args()
    try:
        download(args.output_dir)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        requests.RequestException,
    ) as error:
        # Requests exception messages can contain signed CDN URLs. Keep those out of logs.
        message = (
            type(error).__name__
            if isinstance(error, requests.RequestException)
            else str(error)
        )
        parser.exit(1, f"Vision extraction failed: {message}\n")


if __name__ == "__main__":
    main()
