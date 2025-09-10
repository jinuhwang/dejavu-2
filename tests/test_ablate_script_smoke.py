import os
import sys
import shutil
import subprocess
import pytest


def test_ablate_reuse_script_dry_run_smoke():
    """
    Smoke-test that the ablation script parses and expands configs in DRY_RUN.
    Ensures it can be invoked from tests without running any training.
    """
    # Skip on platforms without bash
    if shutil.which("bash") is None:
        pytest.skip("bash not available; skipping ablation script smoke test")

    repo_root = os.path.dirname(os.path.dirname(__file__))
    script = os.path.join(repo_root, "scripts", "ablate_reuse.sh")
    assert os.path.exists(script), "scripts/ablate_reuse.sh not found"

    env = os.environ.copy()
    # Ensure no external logging side effects and minimal work
    env.update(
        {
            "DRY_RUN": "1",  # do not execute python training
            "DEBUG": "1",  # force debug config path and avoid loggers/callbacks
            "LOGGER": "none",
            "ACCELERATOR": "cpu",
            "DEVICES": "1",
            "SEEDS": "123",
            "START_PHASE": "1",
            "END_PHASE": "1",
            # keep data group simple; used only for config reads in inline Python
            "DATA": "default",
        }
    )

    # Run the script via bash to ensure consistent behavior
    proc = subprocess.run(
        ["bash", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    # The script should exit cleanly in DRY_RUN
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, "ablate_reuse.sh exited with non-zero status"

    out = proc.stdout
    # Basic sanity: should print phase header and one seed command expansion
    assert "==== Phase 1 ====" in out
    assert ">>> Running group=" in out
    assert "CMD: python" in out
