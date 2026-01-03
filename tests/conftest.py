import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())
