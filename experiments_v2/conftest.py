"""Expose experiments_v2 modules to pytest without installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
