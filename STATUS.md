# STATUS — 自动选片训练当前进度

> 最近更新：2026-11-11（团内取舍缺口的候选方向逐一检验）。本文每会话末重写；数字真源 → [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) §5，仪表盘为既有 val 快照。
> 用户亲笔真源 → [docs/HUMAN.md](docs/HUMAN.md)；目标 → [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) §3/§4；方法论 → [docs/METHODOLOGY.md](docs/METHODOLOGY.md)；证据 → [EXECUTION-LOG.md](EXECUTION-LOG.md)。

## 距目标（≥3★ 选出与 0–3★ 准确打分）

本次只评 v2 val 四事件，旧星为代理真值。随机 AP 使用有限样本解析期望。四档 = 0／1／2／≥3；4–5★ 暂并入 ≥3 档。

| 方法（val） | AP≥3 | AP 缺口闭合 | 四档平均召回 |
|---|---:|---:|---:|
| 随机 / 均匀随机四档 | 0.1122 | 0 | 0.2500 |
| 完美排序 / 分档上限 | 1.0000 | 1.0000 | 1.0000 |
| 名次和直排（固定排序基线） | 0.1638 | 0.0581 | — |
| 名次和 + 类别均衡分档（校准对照） | 0.1639 | 0.0582 | 0.3156 |
| 联合视觉头：事件等权 | 0.1647 | 0.0591 | 0.2496 |
| 联合视觉头：事件内类别等权 | 0.1660 | 0.0605 | 0.2900 |

- **整体未达标**，本会话没有新 val 成绩：所有新方法都停在 train 闸，val/test 未打开。
- **差距已定位**：最大可回收量是同段同对象不同构图/时机的取舍（train OOF 理想段内相似组重排 AP≥3 0.206→0.337）。
- **该缺口在所有可得表征与监督粒度上都无可迁移信号**（KNOWLEDGE §5.1.2 小结）：冻结 DINO patch 比较器 0.514、8 个外部零样本美学/画质模型最高 0.552（与单图头 0.565 持平）、事件内跨段学习 0.511；换成用户盲评真值也不变（标签本身可重复，原星↔盲评 0.712）。
- **能继续变好的只有整场粗排序**：独立事件数 7→13 时留出 AP≥3 +0.0115 [+0.0027,+0.0210]，团内准确率不变。

## 当前位置

- **目标与方法分开**：核心目标只放 REQUIREMENTS §3/§4；时间线/邻域/比较都是方法；“取末张”评分规则禁止。
- **统一评估**：`photo-rating-v2-20261014`；历史每团一张指标标 legacy。
- **固定基线**：`train/rating_baseline.py` 两臂，train OOF 选参后只评 val；名次和仅作对照。
- **输入**：`display-plane-v3` 修复 CLS；仓外修复 patch 网格（train+val）与保持宽高比预览（train）缓存，布局 → data/README。旧 DB 的 CV/patch 不可混用。
- **v2 是开发留出**：test 已被历史开发接触；独立终考只能在未来新拍摄事件上完成。

## 下一步与停止条件

1. **停止团内取舍的方法搜索**：冻结/微调/外部零样本表征、比较式/单图/事件内学习、干净真值都已有边界明确的负结论（墓碑表）。重开须带来能区分同场景不同构图/时机的新信息源，而不是同类表征/头的再组合。
2. **需要用户裁定的方向转换**（见等待项）：把自动化定位为“整场粗排 + 相似组聚合供人工快速择一”，团内终选交给人。若接受，下一步实验应针对这一产品形态的验收：例如相似组聚合质量、组级粗排 AP、每组保留一张时的召回/纯度曲线（不是每团一张的旧结构，而是让人在组内选）。
3. **可继续投入的杠杆**：新拍摄事件持续入库（整场粗排序未饱和）；新事件同时充当独立终考。
4. **停止纪律**：不据 val/test 回调参数，不恢复每团一张作为自动终选，不为验收发起标注。

## 等待用户项

- 是否接受“团内终选交给人、模型负责整场粗排与相似组聚合”的产品定位（这改变 REQUIREMENTS §3 中“0–3★ 全部自动准确打分”的落地方式，需用户裁定后再写入 REQUIREMENTS）。
- 若仍希望自动团内取舍，需要用户提供新的信息源：例如对少量同场景对写明取舍理由（构图 / 路人时机 / 光线 / 清晰），用于检验是否存在可分解的子因素。
- 全局精品池/跨组题仍冻结；东莞时钟异常与 2025 批 sidecar 标签载体见 data/BATCHES.md。

## 产物、参数与仓库状态

- 划分：`data/split-v2-20260924.json`；输入：`train/out/recovery_work/corrected_features.npz`；仓外缓存：`D:/PhotoDB/dataset/patches-display-plane-v3-dinov3s-20261111`（14.4GB）、`D:/PhotoDB/dataset/preview1024-display-plane-v3-train-20261111`（18GB）。
- 基线：`train/out/rating_baseline_20261014_{event,balanced}/`；统一评估 `audit/out/rating_protocol_val_20261014/`。
- 本会话：`audit/out/within_scene_audit_20261111/`、`train/out/scene_comparator_20261111{,_steps300}/`、`audit/out/golden_label_check_20261111/`、`audit/out/event_learning_curve_20261111/`、`audit/out/zero_shot_20261111/`、`audit/out/within_event_transfer_20261111/`；命令与数字 → LOG 2026-11-11。
- HUMAN、冻结 reviews、数据库、原片、标签与 split 未写入。

## 新会话起点

按 AGENTS 阅读纪律读本页 → HUMAN 全文 → REQUIREMENTS；先看等待项中的产品定位裁定，再决定是做“粗排 + 组聚合”的验收设计，还是收集新的团内取舍信息源。新方向前读 METHODOLOGY、KNOWLEDGE §5.1.2 小结与墓碑表、SUPERVISION 先验。
