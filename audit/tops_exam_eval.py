"""全局精选开发回归入口：模型自主选团代表，固定预算，直接盲评严格分同/跨事件。"""
from __future__ import annotations

import argparse
from pathlib import Path

from protocol_eval import evaluate


def main() -> None:
    """委托统一协议生成全链路旧标签代理与盲评报告，不再用人工团顶挑候选。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="protocol_eval freeze 生成的目录")
    parser.add_argument("--scores", nargs="+", required=True, help="name=path；必须覆盖全部 test 照片")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.snapshot, args.scores, args.out, local_only=False)


if __name__ == "__main__":
    main()
