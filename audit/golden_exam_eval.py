"""金标准局部考卷入口：统一处理完整覆盖、组上下文、人机同分母及模型等分。"""
from __future__ import annotations

import argparse
from pathlib import Path

from protocol_eval import evaluate


def main() -> None:
    """在冻结快照上评估局部分数，保存不可覆盖的 JSON 与 Markdown 报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="protocol_eval freeze 生成的目录")
    parser.add_argument("--scores", nargs="+", required=True, help="name=path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.snapshot, args.scores, args.out, local_only=True)


if __name__ == "__main__":
    main()
