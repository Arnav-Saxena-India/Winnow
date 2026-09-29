"""Timings, and the offline guard that makes the network counter honest.

``assert_offline()`` patches the socket layer so any attempt to reach a
non-loopback address raises ``NetworkAttempt`` and is counted. Loopback is
allowed: a model served on this machine is on-device. The UI's "net 0" is
``attempts()``, so the counter is enforced rather than decorative.
"""
from __future__ import annotations

import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from winnow.types import NetworkAttempt

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_attempts = 0


def attempts() -> int:
    return _attempts


def _host(address) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return ""          # AF_UNIX paths, pipes: local by definition


def _refuse(host: str):
    global _attempts
    _attempts += 1
    raise NetworkAttempt(f"network access to {host!r} attempted inside assert_offline()")


@contextmanager
def assert_offline():
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if _host(address) not in _LOOPBACK:
            _refuse(_host(address))
        return orig_connect(self, address)

    def connect_ex(self, address):
        if _host(address) not in _LOOPBACK:
            _refuse(_host(address))
        return orig_connect_ex(self, address)

    def getaddrinfo(host, *a, **kw):
        h = host.decode() if isinstance(host, bytes) else str(host or "")
        if h not in _LOOPBACK:
            _refuse(h)
        return orig_getaddrinfo(host, *a, **kw)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.getaddrinfo = getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = orig_connect, orig_connect_ex
        socket.getaddrinfo = orig_getaddrinfo


def install_offline_guard() -> None:
    """For process-pool workers: enter the guard for the life of the process."""
    assert_offline().__enter__()


@dataclass
class Telemetry:
    records: list[dict] = field(default_factory=list)

    def record(self, stage: str, backend: str, device: str, ms: float, **extra) -> None:
        self.records.append({"stage": stage, "backend": backend, "device": device,
                             "ms": round(ms, 1), **extra})

    @contextmanager
    def timed(self, stage: str, backend: str, device: str, **extra):
        t = time.perf_counter()
        yield
        self.record(stage, backend, device, (time.perf_counter() - t) * 1000, **extra)

    def total_ms(self, stage: str) -> float:
        return sum(r["ms"] for r in self.records if r["stage"] == stage)
