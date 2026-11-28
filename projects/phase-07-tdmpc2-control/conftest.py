import os
import sys

_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_PROJ_DIR, "..", "..", "src"))
_P5_BENCH = os.path.abspath(os.path.join(_PROJ_DIR, "..", "phase-05-control-benchmark"))
for p in (_PROJ_DIR, _SRC_DIR, _P5_BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)
