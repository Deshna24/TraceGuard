"""TRACEGUARD-X: isolated experimental extension of the TRACEGUARD research project.

All new experimental artifacts live under ``traceguard_x/``; frozen v4.1
artifacts under ``traceguard/outputs/`` and ``traceguard/data/`` are
read-only. Importing this package inserts the ``traceguard/`` directory at
the front of ``sys.path`` so the canonical pipeline modules (``src.config``,
``src.preprocessing``, ...) resolve exactly as the frozen pipeline expects.
"""

import sys
from pathlib import Path

_TRACEGUARD_PKG_DIR = Path(__file__).resolve().parent.parent / "traceguard"

if str(_TRACEGUARD_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACEGUARD_PKG_DIR))
elif sys.path.index(str(_TRACEGUARD_PKG_DIR)) != 0:
    # Ensure the canonical package wins over any same-named module (e.g. the
    # legacy root-level ``src/`` directory) in this process.
    sys.path.remove(str(_TRACEGUARD_PKG_DIR))
    sys.path.insert(0, str(_TRACEGUARD_PKG_DIR))

__version__ = "0.1.0"
