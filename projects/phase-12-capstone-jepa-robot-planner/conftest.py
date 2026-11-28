"""Phase 12 毕业项目 pytest 路径配置。"""

import os
import sys

_PROJ = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_PROJ)), "src")
for _p in (_PROJ, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
