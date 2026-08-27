"""
golden_retest_eval.py — 人类线复测分析：同卷重标 vs 原始标星的一致性

输入：
  - D:/PhotoDB/dataset/golden_retest/（用户重标后的 R###_X，内嵌/sidecar 星级）
  - golden_retest/golden_retest_key.csv（R###_X ↔ 原 gid/anon/fingerprint 映射）
  - golden_exam/golden_exam_readback.tsv（原始标星明细：gid/anon/fingerprint/user_star）

口径（与考卷评估对齐，golden_exam_eval 的人类版）：
  - 逐组：原始轮胜者集 = user_star 最大且 >0 的成员；复测轮同理；
  - 双决胜组（两轮都有唯一胜者）：冠军一致 = 两轮胜者同一张（按 fingerprint 比）；
  - 前二命中 = 复测胜者的原始星 ≥ 原始轮次高星（并列冠军天然满足冠军位）；
  - 任一轮 tie/tie0 的组单列，不进双决胜分母。

输出：控制台 + golden_retest/retest_report.md。
这是"达到人类线即可用"判据的正式分母（替换跨年上界 0.66-0.73/0.93）。

用法（仓根 D:/Git/PhotoViewer 下）：
    PYTHONUTF8=1 Tools/.venv/Scripts/python.exe Training/audit/golden_retest_eval.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Training/
sys.path.insert(0, str(ROOT / "audit"))
from golden_star_readback import read_rating  # noqa: E402

RETEST_DIR = Path("D:/PhotoDB/dataset/golden_retest")
ORIG_TSV = Path("D:/PhotoDB/dataset/golden_exam/golden_exam_readback.tsv")


def winners(stars: dict[str, int]) -> tuple[str, list[str]]:
    """stars: fp → star。返回 (status, 胜者 fp 列表)。"""
    top = max(stars.values())
    if top == 0:
        return "tie0", []
    ws = [f for f, s in stars.items() if s == top]
    return ("win" if len(ws) == 1 else "tie"), ws


def main() -> int:
    key = list(csv.DictReader(open(RETEST_DIR / "golden_retest_key.csv", encoding="utf-8-sig")))
    r2orig = {(r["r_gid"], r["r_anon"]): r["fingerprint"] for r in key}
    orig_gid = {r["fingerprint"]: r["orig_gid"] for r in key}

    # 复测轮星级：R 卷读回 → fingerprint
    re_stars: dict[str, dict[str, int]] = defaultdict(dict)
    n_read = 0
    for (rg, ra), fp in r2orig.items():
        cands = [f for f in RETEST_DIR.glob(f"{rg}_{ra}.*") if f.suffix.lower() != ".xmp"]
        if not cands:
            continue
        re_stars[orig_gid[fp]][fp] = read_rating(cands[0])
        n_read += 1
    # 原始轮星级：readback tsv → fingerprint
    orig_stars: dict[str, dict[str, int]] = defaultdict(dict)
    for ln in open(ORIG_TSV, encoding="utf-8"):
        if ln.startswith("gid"):
            continue
        p = ln.rstrip("\n").split("\t")
        orig_stars[p[0]][p[2]] = int(p[3])
    print(f"复测读回 {n_read} 张；原始轮 {len(orig_stars)} 组 / 复测轮 {len(re_stars)} 组")

    stat = Counter()
    n_agree = n_top2 = n_dec = 0
    lines = []
    for gid in sorted(orig_stars):
        if gid not in re_stars:
            continue
        o_status, o_win = winners(orig_stars[gid])
        r_status, r_win = winners(re_stars[gid])
        o_top = max(orig_stars[gid].values())
        o_second = sorted(orig_stars[gid].values(), reverse=True)[1] \
            if len(orig_stars[gid]) > 1 else 0
        if o_status != "win" or r_status != "win":
            stat[f"skip(原{o_status}/复{r_status})"] += 1
            continue
        n_dec += 1
        ow, rw = o_win[0], r_win[0]
        agree = ow == rw
        top2 = orig_stars[gid][rw] >= o_second
        n_agree += agree
        n_top2 += top2
        lines.append(f"{gid}: 原胜 {ow[:8]}(★{o_top}) 复胜 {rw[:8]}"
                     f"(原★{orig_stars[gid][rw]}) {'✓' if agree else '✗'}"
                     f"{' / 前二✓' if top2 else ' / 前二✗'}")

    n = max(n_dec, 1)
    head = [
        "# 团内人类线复测报告（同卷重标）",
        "",
        f"- 双决胜组 {n_dec}；跳过（任一轮 tie/全0）{sum(stat.values())}：{dict(stat)}",
        f"- **冠军一致率 = {n_agree}/{n_dec} = {n_agree / n:.3f}**"
        f"（对照：跨年上界 0.66-0.73；模型系综 0.47 / 规则候选 0.53）",
        f"- **前二一致率 = {n_top2}/{n_dec} = {n_top2 / n:.3f}**"
        f"（对照：跨年上界 0.93；模型 0.93）",
        "",
        "## 逐组明细",
        *lines,
    ]
    report = "\n".join(head) + "\n"
    (RETEST_DIR / "retest_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"[OK] {RETEST_DIR / 'retest_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
