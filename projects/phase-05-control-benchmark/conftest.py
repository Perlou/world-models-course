import os
import sys

_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_PROJ_DIR, "..", "..", "src"))
for p in (_PROJ_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
