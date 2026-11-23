"""
课程检查脚本
============

两个用途：

1. 检查各章的 README.md 和 CONCEPT.md 是否符合 CLAUDE.md 里的规范（默认）
2. 以冒烟模式（WM_QUICK=1）运行某一章的全部讲义（--run）

用法：
    uv run python scripts/check_course.py                 # 检查全部章节的文档
    uv run python scripts/check_course.py phase-03        # 只检查某一章（可写多个）
    uv run python scripts/check_course.py --run phase-01  # 冒烟运行某一章的讲义

退出码：有 ❌ 时为 1，否则为 0。⚠️ 和 ℹ️ 不影响退出码。
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# CONCEPT.md 的最少行数：P1 和 P12 为 800，其余为 1200（见 CLAUDE.md）
MIN_LINES_DEFAULT = 1200
MIN_LINES_LIGHT = {"phase-01": 800, "phase-12": 800}

README_SECTIONS = ["🧭", "📚", "🧠", "🎯", "📖", "💻", "✅"]

# CONCEPT.md 必须出现的标题：(说明, 匹配标题行的正则)
CONCEPT_SECTIONS = [
    ("目录", r"^## 目录"),
    ("0. 本章导读", r"^## 0\. 本章导读"),
    ("0.1 本章要回答的问题", r"^### 0\.1 "),
    ("0.2 一图总览", r"^### 0\.2 "),
    ("0.3 前置知识检查", r"^### 0\.3 "),
    ("0.4 概念与讲义对照表", r"^### 0\.4 .*对照"),
    ("0.5 符号表", r"^### 0\.5 .*符号"),
    ("方法对比与选型", r"^## \d+\. 方法对比"),
    ("论文精读卡片", r"^## \d+\. 论文精读"),
    ("实现陷阱与调试清单", r"^## \d+\. 实现陷阱"),
    ("常见问题（FAQ）", r"^## \d+\. 常见问题"),
    ("自测题", r"^## \d+\. 自测题"),
    ("小结与下章预告", r"^## \d+\. 小结"),
    ("附录 A：术语表", r"^## 附录 A"),
    ("附录 B：延伸阅读", r"^## 附录 B"),
]

LESSON_RE = re.compile(r"`(\d{2}-[a-z0-9-]+\.py)`")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")


class Report:
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def info(self, msg: str):
        self.infos.append(msg)

    def show(self):
        mark = "❌" if self.errors else ("⚠️ " if self.warnings else "✅")
        print(f"\n{mark} {self.name}")
        for m in self.errors:
            print(f"   ❌ {m}")
        for m in self.warnings:
            print(f"   ⚠️  {m}")
        for m in self.infos:
            print(f"   ℹ️  {m}")


# ==================== Markdown 工具 ====================


def split_code(lines: list[str]) -> list[bool]:
    """返回每一行是否在代码块里（含围栏行本身）。"""
    in_code = False
    flags = []
    for line in lines:
        if line.lstrip().startswith("```"):
            flags.append(True)
            in_code = not in_code
        else:
            flags.append(in_code)
    return flags


def github_slug(text: str) -> str:
    """模仿 GitHub 生成标题锚点的规则。"""
    text = re.sub(r"`([^`]*)`", r"\1", text)  # 去掉行内代码的反引号
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # 链接只留文字
    text = text.strip().lower()
    # 只保留：字母数字（含中文）、空格、连字符、下划线
    text = "".join(ch for ch in text if ch.isalnum() or ch in " -_")
    return text.replace(" ", "-")


def collect_anchors(lines: list[str], in_code: list[bool]) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line, code in zip(lines, in_code):
        if code:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        slug = github_slug(m.group(2))
        n = seen.get(slug, 0)
        anchors.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    return anchors


def strip_inline_code(line: str) -> str:
    return re.sub(r"`[^`]*`", "", line)


def section_slice(lines: list[str], in_code: list[bool], pattern: str) -> list[str]:
    """取出匹配 pattern 的二级标题所在的整节内容（到下一个二级标题为止）。"""
    start = None
    for i, (line, code) in enumerate(zip(lines, in_code)):
        if code:
            continue
        if start is None and re.match(pattern, line):
            start = i
        elif start is not None and line.startswith("## "):
            return lines[start:i]
    return lines[start:] if start is not None else []


# ==================== README 检查 ====================


def lessons_in_readme(readme: Path) -> list[str]:
    if not readme.exists():
        return []
    found = LESSON_RE.findall(readme.read_text(encoding="utf-8"))
    return sorted(set(found))


def check_readme(phase_dir: Path, rep: Report):
    readme = phase_dir / "README.md"
    if not readme.exists():
        rep.error("缺少 README.md")
        return
    text = readme.read_text(encoding="utf-8")
    for emoji in README_SECTIONS:
        if not re.search(rf"^## {emoji}", text, flags=re.M):
            rep.error(f"README.md 缺少「## {emoji} …」一节")
    if "**状态**" not in text:
        rep.error("README.md 顶部缺少「> **状态**：…」这一行")
    lessons = lessons_in_readme(readme)
    if not lessons:
        rep.error("README.md 里没有列出任何讲义文件（形如 `01-xxx.py`）")
    else:
        existing = [name for name in lessons if (phase_dir / name).exists()]
        rep.info(f"README：列出 {len(lessons)} 节讲义，已写 {len(existing)} 节")


# ==================== CONCEPT 检查 ====================


def check_concept(phase_dir: Path, rep: Report):
    concept = phase_dir / "CONCEPT.md"
    if not concept.exists():
        rep.warn("CONCEPT.md 还没写")
        return
    text = concept.read_text(encoding="utf-8")
    lines = text.split("\n")
    in_code = split_code(lines)
    prose = [line for line, code in zip(lines, in_code) if not code]

    # 1. 篇幅
    phase_key = phase_dir.name[:8]
    min_lines = MIN_LINES_LIGHT.get(phase_key, MIN_LINES_DEFAULT)
    n_lines = len(lines)
    if n_lines < min_lines:
        rep.error(f"CONCEPT.md 只有 {n_lines} 行，要求不少于 {min_lines} 行")

    # 2. 必备章节
    for label, pattern in CONCEPT_SECTIONS:
        if not any(re.match(pattern, line) for line in prose):
            rep.error(f"CONCEPT.md 缺少章节：{label}")

    # 3. 每节讲义都要在对照表里出现
    for name in lessons_in_readme(phase_dir / "README.md"):
        if name not in text:
            rep.error(f"CONCEPT.md 没有提到讲义 {name}（对照表必须覆盖每一节讲义）")

    # 4. 代码围栏、<details> 配对
    fences = sum(1 for line in lines if line.lstrip().startswith("```"))
    if fences % 2:
        rep.error(f"代码围栏 ``` 不配对（共 {fences} 个）")
    n_open, n_close = text.count("<details>"), text.count("</details>")
    if n_open != n_close:
        rep.error(f"<details> 与 </details> 数量不一致（{n_open} / {n_close}）")

    # 5. 公式定界符
    display_open = False
    display_start = 0
    for i, (line, code) in enumerate(zip(lines, in_code), start=1):
        if code:
            continue
        stripped = line.strip()
        body = strip_inline_code(line).replace("\\$", "")
        if stripped == "$$":
            display_open = not display_open
            display_start = i
            continue
        if display_open:
            continue
        if body.count("$$") % 2:
            rep.error(f"第 {i} 行：$$ 不配对")
            continue
        if body.replace("$$", "").count("$") % 2:
            rep.error(f"第 {i} 行：行内公式的 $ 不配对")
        # 表格行里的公式不能含竖线
        if stripped.startswith("|"):
            for math in re.findall(r"\$([^$]+)\$", body):
                if "|" in math:
                    rep.error(f"第 {i} 行：表格里的公式含竖线「|」，请改用 \\mid 或 \\Vert")
                    break
    if display_open:
        rep.error(f"第 {display_start} 行开始的 $$ 独立公式没有闭合")

    # 6. 链接：页内锚点和相对路径
    anchors = collect_anchors(lines, in_code)
    for i, (line, code) in enumerate(zip(lines, in_code), start=1):
        if code:
            continue
        for target in LINK_RE.findall(strip_inline_code(line)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                if target[1:] not in anchors:
                    rep.error(f"第 {i} 行：页内链接 {target} 找不到对应标题")
                continue
            path = target.split("#")[0]
            if path and not (phase_dir / path).exists():
                rep.error(f"第 {i} 行：相对链接 {target} 指向的文件不存在")

    # 7. 陷阱表、FAQ、自测题的数量
    pitfalls = section_slice(lines, in_code, r"^## \d+\. 实现陷阱")
    rows = [l for l in pitfalls if l.strip().startswith("|") and not re.match(r"^\s*\|[\s:|-]+\|\s*$", l)]
    n_pitfalls = max(len(rows) - 1, 0)  # 减去表头
    if pitfalls and n_pitfalls < 8:
        rep.error(f"陷阱清单只有 {n_pitfalls} 行，要求不少于 8 行")

    faq = section_slice(lines, in_code, r"^## \d+\. 常见问题")
    n_faq = sum(1 for l in faq if re.match(r"^### Q\d+", l))
    if faq and n_faq < 10:
        rep.error(f"FAQ 只有 {n_faq} 条（标题需写成「### Q1：…」），要求不少于 10 条")

    quiz = section_slice(lines, in_code, r"^## \d+\. 自测题")
    n_quiz = sum(1 for l in quiz if "<details>" in l)
    if quiz and n_quiz < 12:
        rep.error(f"自测题只有 {n_quiz} 道带折叠答案的题，要求不少于 12 道")

    # 8. 统计信息
    n_todo = text.count("📌 待回填")
    n_unsure = text.count("待核对")
    n_display = sum(1 for line, code in zip(lines, in_code) if not code and line.strip() == "$$") // 2
    rep.info(
        f"CONCEPT：{n_lines} 行，独立公式 {n_display} 个，陷阱 {n_pitfalls} 行，"
        f"FAQ {n_faq} 条，自测 {n_quiz} 题，待回填 {n_todo} 处，待核对 {n_unsure} 处"
    )


# ==================== 讲义冒烟运行 ====================


def run_lessons(phase_dir: Path) -> bool:
    lessons = sorted(phase_dir.glob("[0-9][0-9]-*.py"))
    if not lessons:
        print(f"{phase_dir.name}：还没有讲义")
        return True
    env = dict(os.environ, WM_QUICK="1")
    all_ok = True
    print(f"\n以冒烟模式运行 {phase_dir.name} 的 {len(lessons)} 节讲义：")
    for lesson in lessons:
        start = time.time()
        proc = subprocess.run(
            [sys.executable, str(lesson)], env=env, cwd=REPO_ROOT, capture_output=True, text=True
        )
        elapsed = time.time() - start
        ok = proc.returncode == 0
        all_ok &= ok
        print(f"   {'✅' if ok else '❌'} {lesson.name:<44} {elapsed:6.1f}s")
        if not ok:
            tail = (proc.stderr or proc.stdout).strip().split("\n")[-12:]
            print("      " + "\n      ".join(tail))
    return all_ok


# ==================== 入口 ====================


def find_phases(selectors: list[str]) -> list[Path]:
    phases = sorted(p for p in SRC.glob("phase-*") if p.is_dir())
    if not selectors:
        return phases
    picked = [p for p in phases if any(p.name.startswith(s) or p.name == s for s in selectors)]
    missing = [s for s in selectors if not any(p.name.startswith(s) for p in phases)]
    for s in missing:
        print(f"❌ 找不到章节：{s}")
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="检查课程文档，或冒烟运行讲义")
    parser.add_argument("phases", nargs="*", help="要检查的章节前缀，如 phase-03；不写则检查全部")
    parser.add_argument("--run", metavar="PHASE", help="以冒烟模式运行该章的全部讲义")
    args = parser.parse_args()

    if args.run:
        phases = find_phases([args.run])
        return 0 if phases and all(run_lessons(p) for p in phases) else 1

    phases = find_phases(args.phases)
    if not phases:
        print("没有找到任何章节目录（src/phase-*）")
        return 1

    failed = 0
    for phase_dir in phases:
        rep = Report(phase_dir.name)
        check_readme(phase_dir, rep)
        check_concept(phase_dir, rep)
        rep.show()
        failed += bool(rep.errors)

    print(f"\n共检查 {len(phases)} 章，{failed} 章有错误。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
