import sys
from pathlib import Path
import os

# Ensure local src/ is importable in tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure Hydra paths resolve (configs/paths/default.yaml expects PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
