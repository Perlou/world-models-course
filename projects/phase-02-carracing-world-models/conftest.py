"""让 pytest 能直接导入本项目模块与仓库 src/ 下的共享工具。"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", "src"))

for p in (REPO_SRC, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
