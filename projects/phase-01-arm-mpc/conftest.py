"""让 tests/ 能直接 import 本项目的模块（目录名带连字符，不能当作包导入）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
