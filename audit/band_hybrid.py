"""局部带条件混合入口：使用冻结快照，分数以组号加指纹保存，禁止跨组覆盖。"""
from __future__ import annotations

import argparse
from pathlib import Path

from protocol_eval import contextual_mix


def main() -> None:
    """解析基础与专家分数并委托统一协议生成局部分数；输出不可覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="protocol_eval freeze 生成的目录")
    parser.add_argument("--base", required=True, help="name=path")
    parser.add_argument("--expert", required=True, help="name=path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contextual_mix(args.snapshot, Path(args.base.split("=", 1)[1]),
                   Path(args.expert.split("=", 1)[1]), args.out)


if __name__ == "__main__":
    main()
