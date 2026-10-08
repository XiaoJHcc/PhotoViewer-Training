# STATUS — 自动选片训练当前进度

> 最近更新：2026-09-29。本文只记**当前状态**，每会话末重写（不追加）。**数字级事实只留指针**：候选成绩 → KNOWLEDGE §5；人类线 → KNOWLEDGE §3；资产规模 → SUPERVISION。
> 认知链：上游真源 → [docs/HUMAN.md](docs/HUMAN.md)；需求与判据 → [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)；习惯与先验 → [docs/SUPERVISION.md](docs/SUPERVISION.md) §0；标注资产 → SUPERVISION §1 起；实测结论与负知识 → [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)；人读叙事 → [docs/analysis-story.md](docs/analysis-story.md)；证据链台账 → [EXECUTION-LOG.md](EXECUTION-LOG.md)；文档规范 → [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)。

## 当前位置

- **核心目标（HUMAN 09-29 正式化）**：**第一期 = 前 3 星**——0→3★ 的技术有效性门槛、雷同去重与代表性选优，把人工细看压到 ~12.5% 且 ≥3★ 不漏 80%；**第一期达成即可考虑产品化**。4–5★ 美学判断为远期目标，与 0–3★ 是同一能力的难度差异，不拆离。详见 [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) §3。
- **阶段**：v2 开发留出时代。事件划分 train 14 / val 4 / test 6（`data/split-v2-20260924.json`，规模 → SUPERVISION §13）；**v2 test 已被开发流程接触，不是独立终考**；最终验收须另留未参与训练与调参的完整新事件。
- **图像输入已修复**（2026-09-26）：HEIF 平面尺寸、JPEG EXIF 方向、DPI 裁切三处错误修复，解码版本 `display-plane-v3`（模型 id `dinov3_vits16_f32_518_v2`），25302 张修复图与特征校验通过；旧 `render518` 缓存**禁止混用**。
- **当前候选**（均为 v2 test 成绩，非终考；**数字真源 → KNOWLEDGE §5**）：修复后局部代表选择（三种子 LoRA 等权）+ 个性化 CLIP 全局排序（冻结 CLIP 嵌入 + 训练事件三折偏好拟合）。局部与全局仍有取舍，未声称达到人类标准。
- **人机同卷对照**（旧考卷 54 双决胜组，数字 → KNOWLEDGE §5；前二口径已收窄，KNOWLEDGE §7）。
- **文档体系**（2026-09-29，待提交）：三章按递进认知链重新划分（需求→先验→标注→验证→结论）；`ANNOTATIONS.md` 改名 `SUPERVISION.md`；HUMAN.md 接入为最高真源（L0 永不偏移 → L1 人机共确 → L2 agent 自主，信源唯一不复述）；目标结构按 HUMAN 正式化；数字级事实已去重为唯一真源 + 指针。明细见 LOG 09-29。

## 下一步与停止条件

1. 继续实验只用修复后缓存与当前候选作对照，不回到旧损坏缓存。
2. 在已有标签上审来源冲突与曝光，比较局部/全局目标的训练配方；直接横评单独加权没有稳定收益时，不原样反复跑。
3. 只有明确某类场景或关系缺失、且现有标签无法回答时，才提出最小补标任务。
4. 胜出候选需同预算、多种子、完整新事件验证；保留率与局部选优一起看，不只看单个 pair accuracy。

## 等待用户项

- 全局池/跨组题的下达指令原文未见于本机会话存档（SUPERVISION §14）：若用户记得当时有补充规则，请补一句以回填。草案属远期（4–5★ 精品层）监督，继续冻结（09-29 裁定）。
- 东莞 6 组时钟异常照片（capture_time 2024-01-22，25东莞-A7C2-NO）仍在库，待剔除（BATCHES 已登记）。

## 已冻结参数与提交状态

- 划分：v2（见上）；输入版本：`display-plane-v3`；横评对：`abs_pairs_v2_sessions_20260924`（规模 → SUPERVISION §10）。
- 上一阶段提交：主仓 `6db0715`（解码修复）；Training `06a93ca` + `fad51bd`（输入修复与叙事）、`6571128`（交接冻结）。文档体系首次重构提交：`e9fcee9`（新增）及其后替换笔；二次重构：`8386f5e`（09-29，认知链分层 + HUMAN 接入 + SUPERVISION 改名 + 阅读纪律 + 事实唯一真源）。

## 新会话起点

- **阅读纪律（强制，用户 2026-09-29 裁定）**：每会话（含压缩续接后）全量重读 [AGENTS.md](AGENTS.md) → 本文 → [docs/HUMAN.md](docs/HUMAN.md) → [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)；探索新方向（新实验/新标注/新架构）前必读 [docs/SUPERVISION.md](docs/SUPERVISION.md)（先验+标注资产）与 [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)（结论+负知识墓碑）。
- 训练输入只用 `D:/PhotoDB/dataset/render518-display-plane-v3` 与 `train/out/recovery_work/corrected_features.npz`；不要把零样本 CLIP+LAION 的负结果误读为个性化 CLIP 适配的结果；不要据已接触的 v2 test 调参。
