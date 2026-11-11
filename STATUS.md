# STATUS — 自动选片训练当前进度

> 最近更新：2026-11-11（同场景差异定位与空间比较器判死）。本文每会话末重写；数字真源 → [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) §5，仪表盘为既有 val 快照。
> 用户亲笔真源 → [docs/HUMAN.md](docs/HUMAN.md)；目标 → [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) §3/§4；方法论 → [docs/METHODOLOGY.md](docs/METHODOLOGY.md)；证据 → [EXECUTION-LOG.md](EXECUTION-LOG.md)。

## 距目标（≥3★ 选出与 0–3★ 准确打分）

本次只评 v2 val 四事件，旧星为代理真值。随机 AP 使用有限样本解析期望，不能与 10-13 单次随机序快照混读。四档 = 0／1／2／≥3；4–5★ 暂并入 ≥3 档，不改原标签。

| 方法（val） | AP≥3 | AP 缺口闭合 | 四档平均召回 |
|---|---:|---:|---:|
| 随机 / 均匀随机四档 | 0.1122 | 0 | 0.2500 |
| 完美排序 / 分档上限 | 1.0000 | 1.0000 | 1.0000 |
| 名次和直排（固定排序基线） | 0.1638 | 0.0581 | — |
| 名次和 + 类别均衡分档（校准对照） | 0.1639 | 0.0582 | 0.3156 |
| 联合视觉头：事件等权 | 0.1647 | 0.0591 | 0.2496 |
| 联合视觉头：事件内类别等权 | 0.1660 | 0.0605 | 0.2900 |

- **整体未达标**：类别均衡视觉头各档召回 = 0.351／0.171／0.200／0.438，仍弱；本会话没有新 val 成绩（比较器未过 train 闸，val/test 均未打开）。
- **差距已定位**：星级方差段间份额仅 2–3%；train OOF 上理想段内相似组（cos≥0.88）重排可把 AP≥3 从 0.206 推到 0.337，连拍内重排只到 0.215。最大缺口 = 同段同对象不同构图/时机的取舍（KNOWLEDGE §5.1.2）。
- **该缺口在冻结 DINO 上读不出**：patch 交叉注意力比较器留出跨档对 0.514，置换对照 0.509、CLS 比较器 0.532，均不如单图 0.565（随机 0.5）；预登记 300 步复核同样失败。已入墓碑。

## 当前位置

- **目标与方法分开**：核心目标只放 REQUIREMENTS §3/§4；时间线/邻域/比较都是方法；“取末张”评分规则禁止。
- **统一评估**：`photo-rating-v2-20261014`（完整事件 AP、各边界 AUC、四档混淆与精度/召回、预算精度/召回/基准率/上限）；历史每团一张指标标 legacy。
- **固定基线**：`train/rating_baseline.py` 两臂（事件等权 / 事件内类别等权），train OOF 选参后只评 val；名次和仅作对照。
- **输入**：`display-plane-v3` 修复 CLS（`corrected_features.npz`）与修复 patch 网格仓外缓存（train+val，与产品 ONNX 逐 token 一致）；旧 DB 的 CV/patch 不可混用。CV 全画面网格与 518 patch 非严格同位。
- **Rating 契约**：只读实际选用文件自身内嵌 XMP；既有冻结特征/标签身份未变。
- **v2 是开发留出**：test 已被历史开发接触；独立终考只能在未来新拍摄事件上完成。

## 下一步与停止条件

1. **冻结表征上的读法已穷尽到此**：不再在冻结 DINO CLS/patch 上换头、换比较结构、换损失或步数；CV 粗聚合、TOPIQ518、有序/累计头、相似邻域、patch 比较器均已有边界明确的负结论。
2. **要回收“段内相似组取舍”缺口，必须改前提**，候选方向（尚未登记，开工前须写预测/最小证伪/判死）：① 让骨干看见差异——同场景对上微调（LoRA）而非冻结读出，注意历史 LoRA“能背不能带”与种子 ±6pt；② 换能看到小物体/边缘的输入——高分辨率裁块或保留全画幅的 patch；③ 换监督——用 golden 团内干净对检验“原星冠军位 ~30% 噪声”是否才是留出贴随机的主因（只做诊断：在 golden 对上测单图与比较器，零标注成本）。③ 成本最低且能区分“表征不行”与“标签不行”，建议先做。
3. **停止纪律**：不据 val/test 回调参数，不恢复每团一张，不用加数据/长训练覆盖已定位差距，不为验收发起标注。

## 等待用户项

- 是否认可上面第 2 条的优先序（先做 golden 干净对诊断，再决定是否付微调成本）。
- 全局精品池/跨组题仍冻结；东莞时钟异常与 2025 批 sidecar 标签载体属独立数据处理事项，见 data/BATCHES.md。

## 产物、参数与仓库状态

- 划分：`data/split-v2-20260924.json`；输入：`display-plane-v3`、`train/out/recovery_work/corrected_features.npz`；patch 缓存 `D:/PhotoDB/dataset/patches-display-plane-v3-dinov3s-20261111`（14.4GB，布局 → data/README）。
- 基线结果：`train/out/rating_baseline_20261014_event/`、`train/out/rating_baseline_20261014_balanced/`；统一评估实跑 `audit/out/rating_protocol_val_20261014/`。
- 本会话：`audit/out/within_scene_audit_20261111/`、`train/out/scene_comparator_20261111/`（正式）、`train/out/scene_comparator_20261111_steps300/`（预算复核）；命令与数字 → LOG 2026-11-11。
- 技术核验：`audit/out/technical_input_20261014/final_comparison/`；样本侧缓存 `D:/PhotoDB/dataset/technical-display-plane-v3-sample-20261026-final`。
- HUMAN、冻结 reviews、数据库、原片、标签与 split 未写入。

## 新会话起点

按 AGENTS 阅读纪律读本页 → HUMAN 全文 → REQUIREMENTS；新方向前读 METHODOLOGY、KNOWLEDGE §5.1.2 与墓碑表、SUPERVISION 先验。先区分“表征读不出”与“标签不可迁移”，再决定是否投入微调或新输入。
