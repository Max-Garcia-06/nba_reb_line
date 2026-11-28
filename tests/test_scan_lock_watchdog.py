"""cron_job.sh scan lock: skip while a scan is fresh, kill one that has hung.

On 2026-11-27 a scan blocked on a dead socket for 4.5h and the lock skipped
every later scan. The lock now records its start time; a holder older than
SCAN_MAX_MIN is killed and the new scan runs.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE_PYTHON = f"""#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then exec {sys.executable} "$@"; fi
exec sleep "${{FAKE_SCAN_SLEEP:-0}}"
"""


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "cron_job.sh", tmp_path / "scripts" / "cron_job.sh")
    (tmp_path / "run_pipeline.py").write_text("")
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(FAKE_PYTHON)
    py.chmod(0o755)
    return tmp_path


def _scan(root, sleep_s, **env):
    e = {**os.environ, "FAKE_SCAN_SLEEP": str(sleep_s), **env}
    return subprocess.Popen(["bash", str(root / "scripts" / "cron_job.sh"), "scan"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)


def _wait_for(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill -0; ps shows it as Z.
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def _start_holder(root):
    lock = root / "logs" / ".scan.lock"
    holder = _scan(root, 300)
    assert _wait_for(lambda: lock.exists() and lock.read_text().strip()), "holder never took the lock"
    return holder, lock


def test_fresh_scan_lock_is_respected(sandbox):
    holder, lock = _start_holder(sandbox)
    try:
        out = subprocess.run(["bash", str(sandbox / "scripts" / "cron_job.sh"), "scan"],
                             capture_output=True, text=True, env={**os.environ, "FAKE_SCAN_SLEEP": "0"}, timeout=30)
        assert "still running; skipping" in out.stdout
        assert holder.poll() is None
    finally:
        holder.kill()
        holder.wait()


def test_hung_scan_is_killed_and_new_scan_runs(sandbox):
    holder, lock = _start_holder(sandbox)
    pid = int(lock.read_text().split()[0])
    # Backdate the lock past SCAN_MAX_MIN, as a 4.5h hang would leave it.
    lock.write_text(f"{pid} {int(time.time()) - 26 * 60}\n")
    try:
        out = subprocess.run(["bash", str(sandbox / "scripts" / "cron_job.sh"), "scan"],
                             capture_output=True, text=True, env={**os.environ, "FAKE_SCAN_SLEEP": "0"}, timeout=60)
        assert "killing it" in out.stdout
        assert "[scan] EXIT 0" in out.stdout
        holder.wait(timeout=15)
        assert _wait_for(lambda: not _alive(pid))
        # The killed holder's EXIT trap must not have deleted the new scan's
        # lock mid-run, and the new scan cleans up its own.
        assert not lock.exists()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()


def test_killed_holder_does_not_remove_a_newer_lock(sandbox):
    holder, lock = _start_holder(sandbox)
    pid = int(lock.read_text().split()[0])
    lock.write_text(f"999999 {int(time.time())}\n")  # someone else's lock now
    # Kill the scan's python child, so the holder's shell finishes and runs its EXIT trap.
    subprocess.run(["pkill", "-TERM", "-P", str(pid)], check=True)
    holder.wait(timeout=15)
    assert lock.read_text().split()[0] == "999999"
