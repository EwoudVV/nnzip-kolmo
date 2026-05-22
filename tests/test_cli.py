"""CLI sanity checks.

The CLI is the user-facing interface (`kolmo c file out` after a `pip install -e`).
Easy to forget to update when the engine evolves underneath; pin the surface
with a smoke test.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke `python -m kolmo ...` from a clean env (skip prime so tests are
    fast; we're not validating compression ratio here)."""
    env = os.environ.copy()
    env["KOLMO_SKIP_PRIME"] = "1"
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, "-m", "kolmo", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_compress_then_decompress_short_form(tmp_path):
    """`kolmo c IN OUT` and `kolmo d OUT IN2` should round-trip."""
    src = tmp_path / "in.txt"
    src.write_bytes(b"the quick brown fox")
    blob = tmp_path / "blob.kmo"
    recovered = tmp_path / "out.txt"

    result = _run(["c", str(src), str(blob)])
    assert result.returncode == 0, result.stderr
    assert blob.exists()
    assert "B ->" in result.stdout

    result = _run(["d", str(blob), str(recovered)])
    assert result.returncode == 0, result.stderr
    assert recovered.read_bytes() == src.read_bytes()


def test_compress_then_decompress_long_form(tmp_path):
    """`kolmo compress` and `kolmo decompress` (long form) work too."""
    src = tmp_path / "in.txt"
    src.write_bytes(b"hi")
    blob = tmp_path / "blob.kmo"
    recovered = tmp_path / "out.txt"

    assert _run(["compress", str(src), str(blob)]).returncode == 0
    assert _run(["decompress", str(blob), str(recovered)]).returncode == 0
    assert recovered.read_bytes() == src.read_bytes()


def test_no_args_prints_usage_and_exits_nonzero():
    result = _run([])
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()


def test_unknown_command_exits_nonzero(tmp_path):
    src = tmp_path / "in"
    src.write_bytes(b"x")
    result = _run(["bogus", str(src), str(tmp_path / "out")])
    assert result.returncode != 0


def test_help_flag_prints_env_var_docs():
    """`kolmo --help` should mention the KOLMO_FIXED env var so users can
    discover the cross-machine mode without having to read source."""
    result = _run(["--help"])
    assert result.returncode == 0
    assert "KOLMO_FIXED" in result.stdout


@pytest.mark.parametrize("op", ["c", "d"])
def test_missing_input_file_fails_cleanly(op, tmp_path):
    """A missing input path should propagate the OSError, not silently
    produce a zero-byte file."""
    result = _run([op, str(tmp_path / "nope"), str(tmp_path / "out")])
    assert result.returncode != 0
