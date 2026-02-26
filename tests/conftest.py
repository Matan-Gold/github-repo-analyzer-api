import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep unit tests deterministic and independent from local shell configuration.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("ENABLE_JUDGE", "0")
