# AGENTS.md — Training

> 模块内手册。跨模块联动(产品算法如何被训练消费、AI 特征如何入库)见主库根 [`AGENTS.md`](../AGENTS.md) §5.4；仓库简介与双仓维护注意见 [README.md](README.md)。

## 模块职责

AI 训练一等模块:从产品仓库(`PhotoViewer/Core`)提取 DINOv3 特征 + CV 网格数据,填充独立训练数据集库,支撑照片排序模型的可行性探针与训练迭代。与产品共用一套提取算法(C# `ProjectReference`);本仓已拆分为独立 git 仓库,嵌套在主库工作区 `PhotoViewer/Training/` 下(双仓维护注意见 [README.md](README.md))。

## 子目录索引

| 目录 | 职责 |
|---|---|
| [DatasetBuilder/](DatasetBuilder/) | C# CLI:清单驱动扫描训练用照片文件夹 → 指纹聚合(RAW/HEIF/JPG 合一)→ EXIF/rating → DINO(原片 CLS + 增强 CLS + patch)→ CV grid → 写入独立数据集库 + 覆盖率报告。深度 `ProjectReference` 共享项目 `PhotoViewer/PhotoViewer.csproj`,提取算法与产品共演进,不允许分叉实现。 |
| [probes/](probes/) | Python 特征可行性探针:`feature_probe.py`(线性探针 + t-SNE,判断 backbone/增强/多视图是否够分)、`spatial_probe.py`(空间感知头判别,复用 `feature_probe` 的配对/split 逻辑)、`abs_probe.py`(§1.5 绝对性探针:abs_set 重标星级 × CLS,事件级留出 LOEO 主口径,复用 `feature_probe` 助手)。`out/` 是每次运行的覆盖式输出(不入库)。 |
| [audit/](audit/) | 数据审计与标注集构建:`data_audit.py`(M1 §1.4 分布审计+阈值校准)、`cluster_mine.py`(相似团挖掘+代表资格审计,M2 代表池地基)、`abs_set_sampler.py`/`abs_pair_sampler.py`(§1.5 重标集)、`m2_pool_builder.py`(M2 标注池生成)、`m2_offset_fit.py`(M2 排序制校准拟合+GATE,支持 --exclude-events 锚点反泄漏)、`m3_pair_gen.py`(M3 训练对生成+split)、`split_inventory.py`(事件覆盖盘点与图片概览)、`evaluation_protocol.py` + `protocol_eval.py`(冻结输入/哈希/严格覆盖/固定预算回归；预算支持局部代表分数与全局排序分数分离)、`golden_exam_eval.py`/`tops_exam_eval.py`(统一协议入口)、`band_hybrid.py`(组内上下文混合入口)、`global_pool_sampler.py`/`global_pool_readback.py`/`global_pair_task.py`(全局精品试点池、读回和跨组直接比较)。`out/` 覆盖式输出(不入库)。 |
| [onnx/](onnx/) | DINOv3 模型导出/校验:`export_dinov3_onnx.py` 从 HuggingFace/ModelScope 权重导出双输出(CLS + patch)ONNX;`verify_onnx_parity.py` 校验 PyTorch vs ONNX 一致性(cosine ≥ 0.999)。改动需同步 `PhotoViewer/Core/AI/DinoModelResources.cs`。 |
| [notebooks/](notebooks/) | `cv_grid_design.ipynb` —— CV 网格设计 PoC(numpy 全量标量验证),已定型归档,不再迭代。 |
| [train/](train/) | 模型训练脚本:`m4_baseline.py`(基线)、`m5_lora.py`(LoRA,含 cvfuse 分块融合头)、`m6_adapt_sim.py`、`m7_extprobe.py`(外部探针)、`m8_ensemble.py`(系综)、`render_cache.py`；`supervision_data.py`/`supervision_ablation.py`(按来源隔离、均衡曝光、事件留出监督对照)、`cache_visual_features.py`(带身份/哈希的可续跑 DINO+CLIP 特征缓存)、`personal_visual_ranker.py`(训练事件三折、来源均衡的个性化视觉排序器)。`out/` 覆盖式输出(不入库)。 |
| [docs/](docs/) | 交接与人读文档:[agent-handover.md](docs/agent-handover.md)(新会话/新 Agent 交接,先读)、[analysis-story.md](docs/analysis-story.md)(人读叙事版)、[route-review-2026-09-21.md](docs/route-review-2026-09-21.md)(需求/评估/监督复核与后续实验建议，纠正历史过强结论)、[transfer-failure-analysis.md](docs/transfer-failure-analysis.md)(历史失败证据归档，适用边界见路线审查)。 |
| [plans/](plans/) | 三期计划文档:plan-3-0 宪法 + plan-3-1(M1 详案)+ plan-3-2/3-3/3-4 契约册,彼此用文件名相对链接。一/二期基建历史与 copilot 原始讨论已归还主仓 [../Plans/](../Plans/)(考古专用;已否定方向收编在 plan-3-0 §3 附录),现行基建状态以根 `AGENTS.md` §5.4 为准。 |
| [data/](data/) | 数据契约文档([data/README.md](data/README.md)):数据集库 schema、与产品 `photos.db` 的对齐关系、`dataset_meta` 版本化约定;入库批次台账([data/BATCHES.md](data/BATCHES.md)):批次源/题材/分组规则/特殊情况,每入库一批更新。数据本体在仓外 `D:\PhotoDB`。 |
| [EXECUTION-LOG.md](EXECUTION-LOG.md) | 执行台账(append-only):每次实验的数据/前提/命令/结果/解读/下一步,跨会话防遗忘;`probes/out/` 每次覆盖写,靠本台账留档历史结论。 |
| [STATUS.md](STATUS.md) | 进度真源(每次会话末重写,不追加):里程碑位置 / 最近 GATE / 下一步 / 等待用户项 / 已冻结参数。开工先读。 |

## 构建 / 运行入口

- **v2 开发协议**：事件配置 `data/split-v2-20260924.json`；`audit/m3_pair_gen.py --split-json ... --derived-splits train`；`audit/abs_pair_gen.py` 按原始/扩充会话隔离横评并合并复测区间。`audit/split_eval.py` 复用统一协议作逐事件与固定预算评估；`train/split_supervision_probe.py` 运行固定 CLS、等步数、三种子的原监督/去派生/直接横评供料对照。当前训练输入解码版本为 `display-plane-v3`，不得把旧 `render518` 与修复缓存混用。

- **构建**:`dotnet build Training.sln`(独立解决方案,仅含 `DatasetBuilder`;**不要**把它加进主 `PhotoViewer.sln`——`DatasetBuilder` 是 `net10.0-windows`,加进跨平台主 sln 会连累 Mac/iOS 头的构建)。
- **运行提取**:`dotnet run --project DatasetBuilder -- --manifest <manifest.json>`(清单驱动,见 [DatasetBuilder/manifest.sample.json](DatasetBuilder/manifest.sample.json))或 `dotnet run --project DatasetBuilder -- <folder>... --scan-only`(只扫描不建库,快速核验批次分布)。
- **姿态导出**:`dotnet run --project DatasetBuilder -- --manifest <manifest.json> [附加文件夹...] --dump-accel <csv>`(逐指纹组读机型 + Sony 0x940F → CSV;三轴恒有,俯仰/横滚仅已校准机型 ILCE-7CM2/ILCE-6700;ILCE-6100 的 0x940F 为全零块=无数据;纯 EXIF 读取不建库、不悬挂)。全库产物 `D:/PhotoDB/dataset/accel_export.csv`。
- **探针**:Python 侧用主库仓根的 `../Tools/.venv`(独立虚拟环境,首轮探针即用它;新建则 `pip install -r probes/requirements.txt`),例如 `../Tools/.venv/Scripts/python.exe probes/feature_probe.py --db D:/PhotoDB/dataset/photos_dataset.db`(从本仓根执行)。

## 对产品的契约

- **单向依赖,产品是唯一真源**:`PhotoViewer/Core`(DINOv3 推理、CV 网格提取、EXIF 读取、增强算法等)是训练侧提取逻辑的唯一真源;`DatasetBuilder` 通过 `ProjectReference` 直接复用,不重新实现、不分叉。
- **训练消费产品,不反向影响产品行为**:训练侧的数据集 schema、探针结论、模型训练不得要求产品代码为训练目的改变用户可见行为;产品功能(如控制栏增强 toggle)独立立项,训练侧只是复用其确定性算法。
- **Python 只吃数据集库 schema**:`probes/` 与产品之间没有代码耦合,仅通过 SQLite 数据集库的表结构契约(见 [data/README.md](data/README.md))交互——产品与提取算法可以自由演进,只要 schema 契约(表名/列名/`model_id`/`cv_grid_spec` 版本化规则)不破坏,探针脚本不需要跟着改。
- **数据在仓外**:训练用原始照片与生成的数据集库位于 `D:\PhotoDB`(仓外,不受仓库大小/清理影响,便携)。仓内只有代码、脚本、计划与文档。

## 文档维护纪律(每次实验收尾必做)

- **[STATUS.md](STATUS.md)**:进度真源,每次会话末**重写**(不追加);开工先读。
- **[EXECUTION-LOG.md](EXECUTION-LOG.md)**:实验台账,**append-only**;每次实验记录数据/前提/命令/结果/解读/下一步。
- **[data/BATCHES.md](data/BATCHES.md)**:每入库一批照片更新批次台账。
- **重大阶段节点**:同步 [docs/agent-handover.md](docs/agent-handover.md) 与 [docs/analysis-story.md](docs/analysis-story.md)。
- **本文件**:目录职责、构建入口、契约变化时同步更新。

## 仓库卫生

- 原始照片与数据集库(SQLite)在**仓外** `D:\PhotoDB`,仓库只含代码、脚本、计划与文档;
- `probes/out/`、`audit/out/`、`train/out/` 为覆盖式输出目录,不入库;
- 数据集库 schema 只许加列(additive),禁止破坏性变更(见 [data/README.md](data/README.md));
- 提取算法与产品仓库共用一套 C# 实现(`ProjectReference`),不允许在训练侧分叉重写。
