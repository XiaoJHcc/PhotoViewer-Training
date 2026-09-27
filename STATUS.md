# STATUS — 自动选片训练当前进度

> 最近更新：2026-09-27。本文只记**当前状态**，每会话末重写（不追加）。
> 需求与判据：[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)；结论与负知识：[docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)；标注资产：[docs/ANNOTATIONS.md](docs/ANNOTATIONS.md)；人读叙事：[docs/analysis-story.md](docs/analysis-story.md)；证据链台账：[EXECUTION-LOG.md](EXECUTION-LOG.md)；文档规范：[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)。

## 当前位置

- **阶段**：v2 开发留出时代。事件划分 train 14 / val 4 / test 6 = 12894/5437/6971 张（`data/split-v2-20260924.json`）；**v2 test 已被开发流程接触，不是独立终考**；最终验收须另留未参与训练与调参的完整新事件。
- **图像输入已修复**（2026-09-26）：HEIF 平面尺寸、JPEG EXIF 方向、DPI 裁切三处错误修复，解码版本 `display-plane-v3`（模型 id `dinov3_vits16_f32_518_v2`），25302 张修复图与特征校验通过；旧 `render518` 缓存**禁止混用**。
- **当前候选**（均为 v2 test 成绩，非终考）：修复后局部代表选择（三种子 LoRA 等权）+ 个性化 CLIP 全局排序（冻结 CLIP 嵌入 + 训练事件三折偏好拟合）——12.5% 预算照片召回 **0.1529**、好代表召回 **0.2219**、跨事件横评 **0.6651**、局部冠军 **0.44**。局部与全局仍有取舍，未声称达到人类标准。
- **人机同卷对照**（旧考卷 54 双决胜组）：混合 0.574 vs 人类 0.704；前二均 0.926（该口径已收窄，见 KNOWLEDGE §4）。
- **文档体系已重构**（2026-09-27）：四层职责制落地；agent-handover / transfer-failure 已分流删除，route-review 归档 docs/reviews/。

## 下一步与停止条件

1. 继续实验只用修复后缓存与当前候选作对照，不回到旧损坏缓存。
2. 在已有标签上审来源冲突与曝光，比较局部/全局目标的训练配方；直接横评单独加权没有稳定收益时，不原样反复跑。
3. 只有明确某类场景或关系缺失、且现有标签无法回答时，才提出最小补标任务。
4. 胜出候选需同预算、多种子、完整新事件验证；保留率与局部选优一起看，不只看单个 pair accuracy。

## 等待用户项

- 全局池 168 张 + 跨组题 168 道：已生成、**暂不发起标注**（草案保留，协议见 ANNOTATIONS §10–§11）。
- 全局池/跨组题的下达指令原文未见于本机会话存档（ANNOTATIONS §13）：若用户记得当时有补充规则，请补一句以回填。

## 已冻结参数与提交状态

- 划分：v2（见上）；输入版本：`display-plane-v3`；横评对：`abs_pairs_v2_sessions_20260924`（26165/1181/2613）。
- 上一阶段提交：主仓 `6db0715`（解码修复）；Training `06a93ca` + `fad51bd`（输入修复与叙事）、`6571128`（交接冻结）。文档体系重构提交：`e9fcee9`（新增）及其后替换笔。

## 新会话起点

- 阅读顺序：[AGENTS.md](AGENTS.md) → 本文；深度知识按 AGENTS 文档地图查 KNOWLEDGE / ANNOTATIONS / REQUIREMENTS / story。
- 训练输入只用 `D:/PhotoDB/dataset/render518-display-plane-v3` 与 `train/out/recovery_work/corrected_features.npz`；不要把零样本 CLIP+LAION 的负结果误读为个性化 CLIP 适配的结果；不要据已接触的 v2 test 调参。
