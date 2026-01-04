"""Convenience runner to ensure local package execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from probe_based_ml_codex.__main__ import main

if __name__ == "__main__":
    main()
