# STATUS — 自动选片训练当前进度

> 最近更新：2026-11-12（构图重心纠偏 + 训练深度追问；本页即下一会话的交接页）。本文每会话末重写；数字真源 → [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) §5，仪表盘为既有 val 快照。
> 用户亲笔真源 → [docs/HUMAN.md](docs/HUMAN.md)；目标 → [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) §3/§4；方法论 → [docs/METHODOLOGY.md](docs/METHODOLOGY.md)；证据与登记 → [EXECUTION-LOG.md](EXECUTION-LOG.md)。

## 距目标（≥3★ 选出与 0–3★ 准确打分）

只评 v2 val 四事件，旧星为代理真值；随机 AP 为有限样本解析期望；四档 = 0／1／2／≥3（4–5★ 并入 ≥3）。

| 方法（val） | AP≥3 | AP 缺口闭合 | 四档平均召回 |
|---|---:|---:|---:|
| 随机 / 均匀随机四档 | 0.1122 | 0 | 0.2500 |
| 完美排序 / 分档上限 | 1.0000 | 1.0000 | 1.0000 |
| 名次和直排（固定排序基线） | 0.1638 | 0.0581 | — |
| 联合视觉头：事件内类别等权 | 0.1660 | 0.0605 | 0.2900 |

- **整体未达标**；本会话无新 val 成绩，所有尝试停在 train 闸，val/test 未打开。
- **最大缺口**：同段同对象不同构图/时机的取舍（train OOF 理想段内相似组重排 AP≥3 0.206→0.337）；现有最好读法在同段相似跨档对上仅 0.565（随机 0.5、上限 1）。
- **整场粗排序**随独立事件数仍在缓慢上升（7→13 事件 AP≥3 +0.0115），新拍摄事件是持续杠杆。

## 本会话确立的事实与决定

- **诊断成立**：既往模型都在学“拍的是什么”（单图分数 25–28% 方差在段间，星级仅 2–3%；CLS 设计上对取景不变）；过去的 patch 用法从未拿掉内容（SUPERVISION #21/#22 已改判）。
- **两试（内容无关布局表示）在原星监督下判死**：52 维几何摘要 + 线性 0.510、逐图 PCA 关系图 + 小卷积 0.524，均不给单图头增量（KNOWLEDGE §5.1.3）。
- **用户裁定（#34）**：DINO 只负责分区与区内容；构图评价器是我们自己的核心训练内容，“表征没直接给出”不等于判死。归因已修正为“原星团内对监督 × 读法”的失败。
- **容量不是瓶颈，监督不是构图目标**：所有学习型读法都背下训练事件（拟合 0.94–0.998），留出 0.51–0.54；30919 对来自 1711 组、79% 含 0★，1–3★ 审美比较仅 6608 对（KNOWLEDGE §5.1.3）。
- **外部构图模型已找到**：FGAesQ（相似序列内细粒度排序，含裁剪子集）、PhotoFramer（7B 构图分）；公开构图标注数据 CADB / GAICD / FGAesthetics（KNOWLEDGE §8 外部资源）。

## 下一会话要做的事（按顺序；协议已在 LOG 2026-11-12 冻结登记）

1. **A：FGAesQ 零样本**（登记条目“FGAesQ 零样本读团内取舍”）：独立环境装官方推理代码 + `hf-mirror.com/yzc002/FGAesQ` 权重（记录 SHA256/commit）→ 逐图模式对 `D:/PhotoDB/dataset/preview1024-display-plane-v3-train-20261111` 打分（核验每张 `preview_sha256`）→ 写 `scores.npz`（`fingerprints` + `fgaesq` 列）→ `train/zero_shot_eval.py --scores ... --out audit/out/<目录>` 按既有闸判读；另报 1–3★ 子集、余弦分带、golden 星差分层诊断。过闸才融合，再过闸才开 val。
2. **A 次臂（可选）**：PhotoFramer 先核单卡可行性（16GB 需量化），只对 1–3★ 跨档对照片与 golden 照片打分，作上限参照。
3. **B：自建构图评价器**（登记条目“自建构图模型路线”）：执行前先把扰动集合、视图采样、B1/B2 先后、是否重导 2048 预览写成新的实验前登记（预测 + 最小实验 + 判死）。B1 = 同图轻/强取景扰动自监督（只用 train 事件像素，两侧同一重采样与 518 方形约定，DINO 冻结）；B2 = CADB/GAICD/FGAesthetics 人工构图标注；评价器输入 = 关系图（分区）+ patch 低维投影（区内容）+ 坐标；先零样本检验本库留出团内对与 golden，再融合/Stage B，最后才考虑星级微调。
4. **每步完成后**：结论入 KNOWLEDGE §5.1.3（或新小节）与墓碑/已修正表，先验 #33/#34 状态回流，story §19.4 后续补写，STATUS 重写。

## 工程注意

- 环境：IQA 环境 `../Tools/.venv-iqa`（torch cu128 + scipy + threadpoolctl）跑本仓 GPU 脚本；单图基线复现须 `threadpool_limits(2)`；HF 直连被阻断，用 `HF_ENDPOINT=https://hf-mirror.com`。
- 复用：`train/layout_probe.py` / `train/layout_map_probe.py` 的对、折、golden、融合、Stage B 与闸门逻辑；关系图缓存 `train/out/layout_map_probe_20261112/maps.npz`、描述缓存 `train/out/layout_probe_20261112_control_random/descriptors.npz`（均为 train 12894 张）。
- 写文件：仓内 `.py` 工作副本为 LF、文档为 CRLF；Windows 上 Python `write_text` 默认写 CRLF，生成脚本时显式 `newline="\n"`；新脚本先 `py_compile` 与缩进检查，运行记录的代码哈希须与入仓版本可对应。

## 等待用户项

- 产品定位（团内终选交给人、模型负责整场粗排与相似组聚合）仍未裁定；用户本次选择继续攻团内取舍（自建构图评价器），该裁定可暂缓。
- 全局精品池/跨组题仍冻结；东莞时钟异常与 2025 批 sidecar 标签载体见 data/BATCHES.md。

## 产物与仓库状态

- 划分 `data/split-v2-20260924.json`；输入 `train/out/recovery_work/corrected_features.npz`；仓外缓存 patch 网格 `D:/PhotoDB/dataset/patches-display-plane-v3-dinov3s-20261111`（train+val，14.4GB）、预览 `D:/PhotoDB/dataset/preview1024-display-plane-v3-train-20261111`（train，18GB）。
- 基线 `train/out/rating_baseline_20261014_{event,balanced}/`；本会话产物 `train/out/layout_probe_20261112*`、`train/out/layout_map_probe_20261112/`。
- HUMAN、冻结 reviews、数据库、原片、标签与 split 未写入。

## 新会话起点

按 AGENTS 阅读纪律：本页 → HUMAN 全文 → REQUIREMENTS §3–§6 → METHODOLOGY（§5 新增“判死对象是组合”）→ KNOWLEDGE §5.1.2/§5.1.3、§6、§8 外部资源 → SUPERVISION §0.2 #21/#22/#33/#34 → LOG 2026-11-12 末三条（深度追问、模型调研、A/B 登记）。然后从“下一会话要做的事”第 1 项开工。
