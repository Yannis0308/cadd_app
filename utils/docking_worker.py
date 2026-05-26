"""Subprocess worker for AutoDock Vina docking.

Runs docking in a clean Python process — no torch, no Streamlit — so that
meeko and vina native extensions load without conflicting with torch's
C++ runtime libraries.

Called by :func:`design_utils.run_vina_docking` via ``subprocess.run``.
"""

import os as _os
import sys as _sys


def _log(msg: str) -> None:
    """Print a diagnostic line to stderr (captured by the parent)."""
    _sys.stderr.write(f"[docking_worker] {msg}\n")
    _sys.stderr.flush()


# Set OpenMP env vars *before* any native libraries are loaded
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
from pathlib import Path

# Ensure the project root is on sys.path
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _main() -> None:
    input_path = _sys.argv[1]
    output_path = _sys.argv[2]

    _log("starting")

    with open(input_path, "rb") as f:
        kwargs = pickle.load(f)
    _log("args loaded")

    # Import meeko/vina BEFORE design_utils (which loads RDKit) so the
    # most "fragile" native libs are loaded first — sometimes the order
    # matters for symbol resolution.
    _log("importing meeko …")
    import meeko  # noqa: F401
    _log("meeko imported")

    _log("importing vina …")
    from vina import Vina  # noqa: F401
    _log("vina imported")

    _log("importing design_utils …")
    from utils.design_utils import _run_vina_docking_impl
    _log("design_utils imported")

    _log("running docking …")
    results, cleaning_info = _run_vina_docking_impl(**kwargs)
    _log("docking complete")

    with open(output_path, "wb") as f:
        pickle.dump((results, cleaning_info), f)
    _log("results written")


if __name__ == "__main__":
    _main()
