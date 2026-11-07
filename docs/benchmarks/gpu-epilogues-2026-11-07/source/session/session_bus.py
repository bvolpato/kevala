#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path


BUS_KEYS = ("DBUS_SESSION_BUS_ADDRESS", "DBUS_SYSTEM_BUS_ADDRESS")


def environment_identity(environment) -> dict:
    return {key: {"present": key in environment, "sha256": hashlib.sha256(environment[key].encode()).hexdigest() if key in environment else None} for key in BUS_KEYS}


def _require(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)


class PrivateSessionBus:
    def __init__(self, evidence: dict) -> None:
        self.evidence = evidence
        self.parent_environment = environment_identity(os.environ)
        self.directory = Path(tempfile.mkdtemp(prefix="kevala-sbus-", dir="/tmp"))
        self.descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self.identity = self._identity()
        self.socket_path = self.directory / "absent.sock"
        _require(len(os.fsencode(self.socket_path)) < 108, "private D-Bus path exceeds Unix socket bound")
        self.address = "unix:path=" + str(self.socket_path)
        evidence.update({"configuration": "unavailable-session-bus-for-fresh-ChromeService-only", "address": self.address, "socketPath": str(self.socket_path), "privateDirectory": self.identity, "parentEnvironmentBefore": self.parent_environment, "parentEnvironmentAfter": None, "serviceEnvironment": None, "checks": [], "cleanup": {"attempted": False, "removed": False}, "globalOrUserBusChanged": False})

    def _identity(self) -> dict:
        named, opened = os.lstat(self.directory), os.fstat(self.descriptor)
        _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode), "private D-Bus directory is not a real directory")
        _require((named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino), "private D-Bus directory identity changed")
        _require(named.st_uid == os.getuid() and stat.S_IMODE(named.st_mode) == 0o700 and self.directory.resolve() == self.directory, "private D-Bus directory owner/mode/path differs")
        return {"path": str(self.directory), "device": str(named.st_dev), "inode": str(named.st_ino), "uid": named.st_uid, "mode": "0700", "symlink": False}

    def _absent(self) -> dict:
        try:
            value = os.stat("absent.sock", dir_fd=self.descriptor, follow_symlinks=False)
            return {"absent": False, "symlink": stat.S_ISLNK(value.st_mode), "mode": str(value.st_mode)}
        except FileNotFoundError:
            return {"absent": True, "symlink": False}

    def _socket_rows(self) -> dict:
        with Path("/proc/net/unix").open("rb") as stream: data = stream.read(2 * 1024 ** 2 + 1)
        _require(len(data) <= 2 * 1024 ** 2, "Unix socket metadata budget exceeded")
        rows = []
        for line in data.splitlines()[1:]:
            fields = line.split(maxsplit=7)
            if len(fields) == 8 and fields[7] == os.fsencode(self.socket_path): rows.append(line.decode("ascii", errors="backslashreplace"))
        return {"source": "/proc/net/unix", "bytesRead": len(data), "maximumBytes": 2 * 1024 ** 2, "matchingRows": rows, "meaning": "exact owned pathname only; no matching bound socket row, not a global listener or continuous trace claim"}

    def check(self, stage: str) -> dict:
        record = {"stage": stage, "status": "running", "directoryBefore": None, "endpointBefore": None, "unixSockets": None, "endpointAfter": None, "directoryAfter": None, "parentEnvironment": environment_identity(os.environ)}
        self.evidence["checks"].append(record)
        try:
            record["directoryBefore"] = self._identity(); _require(record["directoryBefore"] == self.identity, "private D-Bus directory changed")
            record["endpointBefore"] = self._absent()
            record["unixSockets"] = self._socket_rows()
            record["endpointAfter"] = self._absent(); record["directoryAfter"] = self._identity()
            _require(record["endpointBefore"]["absent"] and record["endpointAfter"]["absent"] and not record["unixSockets"]["matchingRows"], "private D-Bus endpoint exists, is a symlink or has a bound socket")
            _require(record["directoryAfter"] == self.identity and record["parentEnvironment"] == self.parent_environment, "private D-Bus identity or parent environment changed")
            record["status"] = "passed"
        except Exception as error:
            record["status"] = "failed"; record["error"] = str(error); raise
        return record

    def service_environment(self, inherited: dict) -> dict:
        _require(environment_identity(inherited) == self.parent_environment, "ChromeService must start from unchanged parent bus environment")
        changed = {**inherited, "DBUS_SESSION_BUS_ADDRESS": self.address}
        _require(changed.get("DBUS_SYSTEM_BUS_ADDRESS") == inherited.get("DBUS_SYSTEM_BUS_ADDRESS"), "system bus environment changed")
        self.evidence["serviceEnvironment"] = {"DBUS_SESSION_BUS_ADDRESS": self.address, "systemBusIdentity": environment_identity(changed)["DBUS_SYSTEM_BUS_ADDRESS"], "scope": "only this newly created ChromeService and its browser descendants; no os.environ or parent/controller/UV/user/system bus mutation"}
        return changed

    def close(self) -> None:
        cleanup = self.evidence["cleanup"]; cleanup["attempted"] = True
        try:
            self.check("after-browser-quit")
            _require(self._identity() == self.identity and not list(self.directory.iterdir()), "private D-Bus directory is changed or not empty; refusing removal")
            os.rmdir(self.directory); cleanup["removed"] = True
        except Exception as error:
            cleanup["error"] = str(error); raise
        finally:
            self.evidence["parentEnvironmentAfter"] = environment_identity(os.environ)
            os.close(self.descriptor)
