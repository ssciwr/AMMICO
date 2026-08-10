#!/usr/bin/env python3
"""Run test_model.py and print peak RSS for the pytest process tree (not collected by pytest)."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tree_rss_bytes(proc: psutil.Process) -> int:
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        os.path.join(os.path.dirname(__file__), "test_model.py"),
        "-v",
        "--tb=line",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("Command:", " ".join(cmd), file=sys.stderr)
    print("CWD:", REPO, file=sys.stderr)
    print("Sampling tree RSS every 0.25s…\n", file=sys.stderr)

    p = subprocess.Popen(cmd, cwd=REPO, env=env)
    peak = 0
    peak_t = 0.0
    t0 = time.perf_counter()
    try:
        root = psutil.Process(p.pid)
    except psutil.NoSuchProcess:
        return p.wait()

    while p.poll() is None:
        try:
            rss = tree_rss_bytes(root)
        except psutil.NoSuchProcess:
            break
        now = time.perf_counter() - t0
        if rss > peak:
            peak = rss
            peak_t = now
        time.sleep(0.25)

    rc = p.wait()
    elapsed = time.perf_counter() - t0

    print("\n--- memory (pytest + children, RSS) ---", file=sys.stderr)
    print(
        f"Peak RSS: {peak / 1024**2:.1f} MiB ({peak / 1024**3:.2f} GiB)",
        file=sys.stderr,
    )
    print(
        f"Peak at ~{peak_t:.1f}s into run (total wall ~{elapsed:.1f}s)", file=sys.stderr
    )
    print(f"Exit code: {rc}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
