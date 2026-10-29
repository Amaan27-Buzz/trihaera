"""Exercise Python introspection entry points used by Windows PowerShell setup."""

import pathlib
import subprocess
import sys


PROBE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "env_probe.py"


def test_environment_probe_paths():
    for action in ("base-executable", "site-packages", "prefix"):
        result = subprocess.run(
            [sys.executable, str(PROBE), action],
            check=True,
            text=True,
            capture_output=True,
        )
        assert pathlib.Path(result.stdout.strip()).exists(), result.stdout
