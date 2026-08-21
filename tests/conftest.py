import sys
from pathlib import Path

# The engine imports as `engine.*` / `contracts.*` from the project root, so tests
# run against exactly the same import paths as production rather than a repackaged
# copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
