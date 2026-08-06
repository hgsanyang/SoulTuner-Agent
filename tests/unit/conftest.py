"""Shared fixtures for the unit suite.

The only thing here is a usable-bash probe. Several tests execute the real
training shell scripts rather than asserting on strings in their source, which
is the right way to test them and also the reason they are host-sensitive: the
first ``bash`` on PATH may be one that cannot open a Windows-style path. When
that happens the script dies before reaching any gate logic, and the test fails
for a reason that has nothing to do with the behaviour under test.

That has already caused a review to see ten failures on a tree the author saw as
green. Probing first turns "mystery failure" into an explicit skip that names
the reason, and — more importantly — stops the suite from being read as green on
a host where these tests never actually ran.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def _probe_bash() -> tuple[str | None, str]:
    """Return (bash path, reason) where a path of None means "do not use it"."""
    bash = shutil.which("bash")
    if not bash:
        return None, "no bash on PATH"
    try:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "probe.sh"
            script.write_text("echo BASH_CAN_READ_THIS\n", encoding="utf-8", newline="\n")
            proc = subprocess.run(
                [bash, script.as_posix()],
                capture_output=True, encoding="utf-8", errors="replace", timeout=60,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"bash at {bash} could not be run: {type(exc).__name__}"
    if "BASH_CAN_READ_THIS" not in (proc.stdout or ""):
        return None, (
            f"bash at {bash} cannot execute a script at a path this platform "
            f"produces (rc={proc.returncode}); a WSL bash cannot read C:/ paths"
        )
    return bash, "ok"


_BASH, _BASH_REASON = _probe_bash()


@pytest.fixture(scope="session")
def usable_bash() -> str:
    """A bash that can actually run a script written to a temp path, or skip."""
    if _BASH is None:
        pytest.skip(f"shell-script tests need a usable bash: {_BASH_REASON}")
    return _BASH
