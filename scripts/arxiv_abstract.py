"""
查论文的标题、作者和摘要
========================

写 CONCEPT.md 时用来核对两件事：arXiv 编号有没有写错；对论文的说法和摘要是否一致。

用法：
    uv run python scripts/arxiv_abstract.py 1805.12114
    uv run python scripts/arxiv_abstract.py 2301.04104 2310.16828     # 一次查多篇
    uv run python scripts/arxiv_abstract.py --short 1805.12114        # 只看标题和作者

数据来自 DataCite（arXiv 论文的 DOI 登记处），比 arXiv 自己的接口更不容易被限流。
只能查到摘要级别的信息；正文里的超参数、表格里的数字查不到。
"""

import json
import sys
import time
import urllib.error
import urllib.request

API = "https://api.datacite.org/dois/10.48550/arXiv.{}"


def fetch(arxiv_id: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(API.format(arxiv_id), timeout=30) as resp:
                return json.load(resp)["data"]["attributes"]
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def show(arxiv_id: str, short: bool) -> bool:
    attrs = fetch(arxiv_id)
    if attrs is None:
        print(f"❌ arXiv:{arxiv_id}  查不到（编号写错了，或者网络有问题）\n")
        return False

    title = " ".join((attrs.get("titles") or [{}])[0].get("title", "?").split())
    creators = [c.get("name", "?") for c in attrs.get("creators") or []]
    authors = "; ".join(creators[:4]) + (f" 等 {len(creators)} 人" if len(creators) > 4 else "")
    print(f"arXiv:{arxiv_id}  ({attrs.get('publicationYear')})")
    print(f"  标题：{title}")
    print(f"  作者：{authors}")
    if not short:
        abstracts = [
            d.get("description", "")
            for d in attrs.get("descriptions") or []
            if d.get("descriptionType") == "Abstract"
        ]
        abstract = " ".join(abstracts[0].split()) if abstracts else "（没有摘要）"
        print(f"  摘要：{abstract}")
    print()
    return True


def main() -> int:
    args = sys.argv[1:]
    short = "--short" in args
    ids = [a for a in args if not a.startswith("--")]
    if not ids:
        print(__doc__)
        return 1
    ok = True
    for i, arxiv_id in enumerate(ids):
        ok &= show(arxiv_id.replace("arXiv:", ""), short)
        if i < len(ids) - 1:
            time.sleep(0.4)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
