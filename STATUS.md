# STATUS — 自动选片训练当前进度

> 最近更新：2026-10-13（距目标差距审计）。本文只记**当前状态**，每会话末重写（不追加）。**数字级事实只留指针**——**唯一例外：下方"距目标"仪表盘为可读性保留数字**（[DOCUMENTATION.md](docs/DOCUMENTATION.md) §5）。
> 认知链：上游真源 → [docs/HUMAN.md](docs/HUMAN.md)；需求与判据 → [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)（§3/§4 已按 10-13 裁定重写）；习惯与先验 → [docs/SUPERVISION.md](docs/SUPERVISION.md) §0；标注资产 → SUPERVISION §1 起；实测结论与负知识 → [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)；方法论 → [docs/METHODOLOGY.md](docs/METHODOLOGY.md)；人读叙事 → [docs/analysis-story.md](docs/analysis-story.md)；证据链台账 → [EXECUTION-LOG.md](EXECUTION-LOG.md)。

## 距目标（第一期：0→3★ 的减量与不漏）

**产品化底线目标 = ≥3★ 准确召回**（不是"众多指标之一"，用户 2026-10-13）。**主判据 = 平均精度 AP**（用户同日裁定，取代"达 80% 召回所需预算"——后者可被预算操纵：随机看够约 78–80% 也能凑到 80% 召回，且是事件构成的函数）。下表为 test 事件宏平均，旧星 ≥3★ 代理真值：

| 选片器 | **AP** | **AP 缺口闭合** | P@12.5% | @12.5% 召回 |
|---|---:|---:|---:|---:|
| 随机（下限） | 0.166 | 0 | 0.162 | 0.126 |
| 上限（完美打分 = 靶） | 1.000 | 1.000 | — | 0.874 |
| **当前候选**（`group_quality`：团代表 LoRA + 团内最高 CLIP 排序） | **0.153** | **−0.015** | 0.164 | 0.139 |
| LoRA 直排（无团去重） | 0.185 | 0.023 | 0.185 | 0.162 |
| CLIP 直排（无团去重） | 0.202 | 0.043 | 0.215 | 0.187 |
| **LoRA+CLIP 名次和（无团去重，未落地）** | **0.204** | **0.046** | **0.237** | 0.207 |

- **读数（重要）**：**当前候选在 test 上不如随机**（AP 缺口闭合 −0.015，6 事件中 5 个低于随机）。机制：`group_quality` 只给"每团代表"排名、非代表压到底部，而约 1/3–2/5 的 ≥3★ 照片不是本团第一张——结构上排不出来。**硬证据是结构上限 0.525（test）/0.614（val）**（KNOWLEDGE §5）。旧的 R80 口径掩盖了这一点。
- **换掉去重结构即转正**：名次和直排 AP 0.204（闭合 +0.046），零训练成本，是当前最该做的一件事。
- **顺序结构（时间窗）是一等机制**：选片单位回到时间序滑窗（HUMAN 机制），`cluster_id`/`is_cluster_top` 降级为诊断工具；顺序信息进训练，但不把"末张更优"之类行为规律规则化（SUPERVISION §0.2 #28）。
- 复算入口：`../Tools/.venv/Scripts/python.exe audit/gap_audit.py`；完整审计 → [docs/reviews/2026-10-13-gap-audit.md](docs/reviews/2026-10-13-gap-audit.md)；数字真源 → KNOWLEDGE §5。
- 事件内 AUC（test）：LoRA 0.546–0.584、个性化 CLIP 0.565–0.629；**0→1 技术闸门 0.570 为全链最弱环**（train 对照 0.722）。

## 当前位置

- **核心目标（2026-10-13 用户重定）**：**产品化底线目标 = ≥3★ 准确召回**——不是"众多指标之一"，而是"能否产品化"的判据本身；第一期 = 0→3★ 的技术有效性门槛、雷同去重与代表性选优（**1–3★ 雨露均沾 = 覆盖性取向**），4–5★ 为远期、不拆离。详见 [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) §3。
- **选片单位（2026-10-13 用户裁定）**：**时间线滑窗局部比选 + 逐层抬举**（HUMAN 机制）是一等机制；"特征余弦 ≥0.88 相似团"是 agent 自造代理，与现实不符——**降级为诊断工具**。顺序信息进训练，但不把行为规律（如"末张更优"）规则化（SUPERVISION §0.2 #28）。
- **验收路径（2026-10-13 用户裁定，两轮）**：**逼近上限即达标路径，不引入人类复测线**（标注人时成本应花在提高模型效果上）；**主指标 = 平均精度 AP + AP 缺口闭合率**（辅助：固定预算精度必带基准率、R80 预算须连同该点纯度），12.5% 只作历史可比点。原 R80 口径已废（可被预算操纵、且是事件构成的函数）。
- **阶段**：v2 开发留出时代。train 14 / val 4 / test 6 事件（`data/split-v2-20260924.json`）；**v2 test 已被开发流程接触，不是独立终考**。
- **数据集时间窗已定型**（2024-01-22 ~ 2026-05-10，无后续事件）→ 终考只能落在**未来新拍摄**上（KNOWLEDGE §4）。
- **图像输入已修复**（2026-09-26）：解码版本 `display-plane-v3`（模型 id `dinov3_vits16_f32_518_v2`），旧 `render518` 缓存禁止混用。**注**：该修复对全局指标的作用在噪声带内（KNOWLEDGE §4），但正确性无争议。
- **文档体系**：2026-10-13 完成"距目标差距审计 + 方法论草案"（[METHODOLOGY.md](docs/METHODOLOGY.md) 新建、[reviews/2026-10-13-gap-audit.md](docs/reviews/2026-10-13-gap-audit.md) 归档），并按用户裁定重写 REQUIREMENTS §3/§4（四项裁定全部落地）。

## 下一步与停止条件

> 顺序按"每单位成本消除多少不确定性"排列（[METHODOLOGY.md](docs/METHODOLOGY.md) §3）。

1. **选片单位回到时间窗**（零训练成本，已裁定）：放弃"每相似团一张"，改为时间序滑窗局部比选 + 近重抑制 + 名次和直排；已验证 test ≥3★ **AP 0.153→0.204**（AP 缺口闭合 −0.015→**+0.046**），落为新的对照基线。
2. **落地新口径**：`cluster_size`/`is_cluster_top` 降级为诊断工具并按下游需求重建；`m3_pair_gen.py` 的窗口对生成、`budget_metrics` 的每团一张同步改；主指标改报 **AP + AP 缺口闭合率**（辅助：固定预算精度必带基准率；R80 预算须连同该点纯度）。
3. **第一期模型主线 = 0→1 技术闸门**（AUC 0.57，减量最大、语义最清、信号最全）：把 0★ 拆成"技术不合格"与"被雷同压低"两类，在时间窗内做相对判别；用全分辨率 CV 网格 + 抖动 + 水平度（当前用 518px 缩略图，正是技术判别被漏掉的原因）。
4. **停止条件**：不再用"再标一批 / 再上数据量 / 再调权重"覆盖整体差距（KNOWLEDGE §6 已三次证死）；不在未定瓶颈环节前跑长训练；单次实验必须带"预测 + 判死条件"（METHODOLOGY §4）；**不为验收而发起标注**（2026-10-13 裁定）。

## 等待用户项

- **REQUIREMENTS §6 四项裁定已于 2026-10-13 全部落地**（① 曲线+上限即靶 / ② 旧 accuracy 剖面与人类线验收退役 / ③ EXIF 不禁不入、0→1 核心信号 = 极相似对中的清晰度 / ④ 废片删除与 0→1 技术判别划清）。**当前无待裁定项**。
- 全局池/跨组题草案仍冻结（属远期 4–5★ 监督）。
- 东莞 6 组时钟异常照片（capture_time 2024-01-22，25东莞-A7C2-NO）仍在库，待剔除（BATCHES 已登记）。

## 已冻结参数与提交状态

- 划分：v2（上）；输入版本：`display-plane-v3`；横评对：`abs_pairs_v2_sessions_20260924`。
- 相邻两次提交：`6276c6b`（适配主仓 P0-M4 像素契约）、`8002d4d`（09-29 文档体系二次重构）；本轮改动见下。
- **本次（2026-10-13，10-14 复核修订）改动文件**：新增 `audit/gap_audit.py`、`docs/METHODOLOGY.md`、`docs/reviews/2026-10-13-gap-audit.md`；更新 `docs/KNOWLEDGE.md`（§1/§3/§4/§5/§6/§7/§8）、`docs/SUPERVISION.md`（§0.2 #28–#31）、`docs/REQUIREMENTS.md`（§3/§4 重写 + §6 改为裁定索引）、`docs/analysis-story.md`（§0 / §17 收窄 / §18 按"审查→发现→修正"重写并预留 18.2–18.4）、`STATUS.md`、`EXECUTION-LOG.md`、`AGENTS.md`、`docs/DOCUMENTATION.md`。**未提交**。
- **10-14 复核修订**：逐项核对文档数字与 `audit/out/gap_audit/` 产物，修正 8 处口径/数字错误（AP 表精度格、随机基线、R80 区间、"非团首"占比口径、"3 倍"塌陷的过度概括、6/24 笔误、≥3★ 占比区间自相矛盾、`§1.5` 末张结论与裁定相冲）；裁定正文归位（REQUIREMENTS §6 由复述改索引，登记留 SUPERVISION §0.2，方法论含义留 METHODOLOGY）。

## 新会话起点

- **阅读纪律（分级，2026-10-13 修订）**：每会话必读 [AGENTS.md](AGENTS.md) → 本文（含上方"距目标"仪表盘） → [docs/HUMAN.md](docs/HUMAN.md)；**探索新方向前**必读 [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)（尤其 §6 墓碑表与重开条件）→ [docs/SUPERVISION.md](docs/SUPERVISION.md) §0；[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) 每会话读 §3–§6。
- 训练输入只用 `D:/PhotoDB/dataset/render518-display-plane-v3` 与 `train/out/recovery_work/corrected_features.npz`；不要把零样本 CLIP+LAION 的负结果误读为个性化 CLIP 适配的结果；不要据已接触的 v2 test 调参。
- **下一位 agent 的第一件事**：读本页仪表盘 → 读 METHODOLOGY §2（三条参照线）→ 明白"当前候选贴随机线、管线结构上限低于目标"，再决定动手做什么。
