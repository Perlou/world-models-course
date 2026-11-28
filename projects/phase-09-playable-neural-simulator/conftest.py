import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "src")
for _p in (_THIS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
