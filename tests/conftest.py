"""
Pytest configuration and shared fixtures for SPILS-Net tests.
"""
import pathlib
import pytest


# ------------------------------------------------------------------ #
# Cleanup: remove .safetensors artefacts left by SPILSNet tests.
# ------------------------------------------------------------------ #

# Directories whose .safetensors files must NEVER be touched by cleanup.
# These hold real trained model checkpoints produced outside of tests.
_EXCLUDE_DIRS = {
    "workspace/surrogate_models",
}


@pytest.fixture(scope="session", autouse=True)
def cleanup_safetensors():
    """
    Session-scoped fixture that runs after all tests complete and deletes
    .safetensors files created during the test session.

    Only files that are NOT inside a protected directory (see _EXCLUDE_DIRS)
    are removed. Directories such as ``workspace/surrogate_models`` contain
    real trained-model checkpoints and are left untouched.

    pytest's built-in ``tmp_path`` already cleans up files written to the
    system temp directory, so this fixture only needs to sweep the workspace.
    """
    yield  # Let all tests run first.

    repo_root = pathlib.Path(__file__).parent.parent
    exclude_abs = {repo_root / d for d in _EXCLUDE_DIRS}

    removed = []
    for f in repo_root.rglob("*.safetensors"):
        # Skip files that live inside any excluded directory.
        if any(f.is_relative_to(excl) for excl in exclude_abs):
            continue
        try:
            f.unlink()
            removed.append(f)
        except OSError:
            pass  # Best-effort; never fail teardown.

    if removed:
        print(f"\n[conftest] Removed {len(removed)} test .safetensors file(s):")
        for f in removed:
            print(f"  {f.relative_to(repo_root)}")
