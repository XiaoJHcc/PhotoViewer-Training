# 交接文档 — DINOv3 照片美学评分（给下一位 Agent）

> **用途**：让接手者在 15 分钟内弄清这个项目的前后因果、当前位置与下一步。阅读顺序：本文 → [../STATUS.md](../STATUS.md)（机器真源·当前进度）→ [analysis-story.md](analysis-story.md)（人读叙事版，**§10 是训练阶段总叙事**）→ [../EXECUTION-LOG.md](../EXECUTION-LOG.md)（证据链台账，append-only）→ [transfer-failure-analysis.md](transfer-failure-analysis.md)（失败矩阵详尽版）。
> **维护**：重大阶段交接时重写；本文版本 = 2026-07-28（提精度专项封板：模型侧空间全关，台阶① 终点 0.766；剩余杠杆全在数据侧）。

---

## 1. 任务目的（需求锚点，先读这个再谈技术）

- **用户与场景**：摄影爱好者（SONY A7C2），多拍选优；单次外拍 ~600 张/天，大量照片高度相似；历史 9418 组已全部人工 0-5 星标注。
- **要替代的工作量**：人工筛一次外拍约 **5 小时**。两类最耗神：① 高度相似照片里选更好的（相对区分）；② 平平无奇里挑氛围优秀的（绝对直觉）。
- **唯一目标（宪法 §0.6）**：让模型真正理解摄影美学，对任意一张照片给出反映其水准的分数。
- **★ 现阶段规则（用户 2026-07-27 定）**：**不考虑任何产品化；模型准确性是唯一成功判据；最低成功限度 = 局部弱实现（三级台阶）**：
  - 台阶①（最低线）：**任意两张高分段照片谁强谁弱**（允许排除"重复压低"污染数据）；
  - 台阶②：相似组内选优（选团顶）；
  - 台阶③：团顶之间排序（高分段美学筛选）。
  产品化是模型做到极限后的下一阶段。在达到目标前不接受逃避方案。
- **评估两条线**（宪法决策 10）：≥3★ 召回 @ Top 12.5%（产品门槛，后置）；0-5 准确性容差三层 exact / ±1 / ≥2 硬错误（目标剖面 exact ≥50% / 飘1 ≤45% / 飘2 ≤5% / ≥3 为 0%）。
- **审美优先级**：构图 > 内容 > 光影/氛围/色彩（影调可后期救，末位）。

## 2. 数据的本质（一切设计约束的来源，压缩版）

- **星级是段内锦标赛逐轮晋级的"生存结果"**：强段内相对信号、几乎为零的跨段绝对信号；**只有排序、无数值意义**（禁相减/平均/回归，只可反推生成过程后按序使用）。
- **清洁度与代表身份绑定**：相似团（段内 cos≥0.88 union-find，2569 团）内落败者的低星含"重复压低"成分；**团顶与孤立照的原始星级干净，是绝对尺唯一合法学习源**。
- **团内精细判别 = 美学任务**（仅"真无差异"才判 tie）；**整数舍入是量化噪声**（Δ=1 降权、评估只在 Δ≥2 硬差异区卡闸门）。
- **0★ 即真 0★**；**评估必须事件级切分**（泄漏 0.99 vs 藏一场 0.50，不切全是假象）；**EXIF 不进打分特征**（反迁移实证）。
- **事件 split 定案**：test={茶博/虎跑/良渚版本}、val={祥睦桥}、train=其余 7 事件；金标准集样本必须落在 test 事件。

## 3. 结论链（每步为何走到下一步）

1. M1 探针三连否定：冻结特征"谁更好"信号跨段零迁移（不是单批/池化/backbone 大小问题）；事件内可学 79-94%（后证为泄漏+内容混杂虚高，干净实测 0.55-0.70）。
2. §1.5 绝对性探针（200 张 ≥3★ 盲评重标）：绝对尺只能由监督引入；新尺方差 = 事件间 15% / 段间 34% / 段内张间 51%。
3. M2 排序制校准（GATE 通过）：~400 张盲评 → 潜变量拟合 `s=g(段内星)+b_seg` → **全库第一把统一的尺**（留出 Δ≥2 0.86-0.90）。
4. M3 训练对（GATE PASS）：window 高权 / global（≥3★团顶）中权 / derived（干净潜分派生）低权；事件整场 split。
5. M4 首考：能背不能带（train 0.977 vs test ~0.52）；EXIF 反迁移弃用；失败映射回决策 8 升级梯。
6. 升级梯 1-3 证死（S@518 零迁移 / L@518 弱脉搏噪声带 / L@1024 不放大）。
7. **梯4 LoRA + 训练阶段 13 实验（2026-07-27）**：精细排序五个维度全部穷尽（见 §4 负知识）；**唯一站住 = 台阶① 高分段粗判断**：系综（lora+lorap+laion+l518 均权）在干净对 **derived Δ≥2 = 0.755**（n=1422，没见过的三场上实测；距 M2 尺自身拟合 0.86-0.90 不远——模型已学到该尺可学大半）。台阶② 团内选优 ≈ 随机（**死因在标签本身**：连拍晋级多为压低+舍入产物，须换 ground truth 而非换模型）；台阶③ 团顶召回 0.244（高星太稀）。

## 4. 已证死方向（负知识，勿重开！）

| 方向 | 死因 | 关键数字 |
|---|---|---|
| 升级梯 1-4（S@518/L@518/L@1024/LoRA） | 能背不能带 | test top-1 0.09-0.24（chance 0.155） |
| patch 空间头（E1） | 未超 CLS 头 | 相似带 0.55 vs 0.59 |
| 事件内自适应（k=25-100 标签微调头） | **保序平移免疫**——整场加减分不改变两两排序 | 一致持平或变差 |
| 监督变体：derived 主 / 相似带专注(E2) / 团等价(E3) / 双团顶(E5) / 至少一端团顶(E6) | 全变差或持平——**全量原配比即最好**（脏对承载数据量+正则；window 对 55% 是"同团双落败者"） | — |
| 外部美学先验（CLIP-L/14+LAION，E4） | **通用美学尺与个人判断几乎不相关** | 全带贴 chance、seg-rho -0.06 |
| 系综（仅粗尺有效） | top-1/召回不动 | 仅 derived 0.676 |
| 多种子 LoRA 扩系综（2026-07-28） | **种子方差 ±6pt**（0.551/0.563/0.675）；加种子 = 稀释幸运成员 | E6 0.701 < 0.755；纯三种子 0.615 |
| 全量微调臂（2026-07-28） | 容量非约束 | 单体 0.596，系综无增益 |
| 405 盲评直督 ×w1/w3/w0.3（2026-07-28） | 318 张独立照片瓶颈（对数≠多样性）；高权干扰单体 | 单体 0.58-0.60；w0.3 自家考卷 0.488 |
| val 贪心系综加权（2026-07-28） | val 单事件太薄，不具系综排序能力 | search 选 laion+l518 → test 0.684 |
| 同 run 快照系综（2026-07-28） | 快照相关性过高 | 0.744 无增益 |
| Python 复刻渲染管线 | 解码/缩放实现差异 → 特征余弦仅 0.90 | **渲染正路 = C# `--dump-render`** |

## 5. 当前位置与接下来（2026-07-28 更新）

**当前最优形态（可复现）**：系综 = `m5_lora`(ep1) + `m5_lora_patch`(ep3) + `m7_extprobe`(LAION) + `m4_l518_cls2`，各分数 z-score 后均权；**台阶① derived dstar==2 = 0.766**（配方带 0.755-0.766 随 lora epoch）。产物 `Training/train/out/m8_best/`；**考卷已固化 = `Training/train/m8_ensemble.py`**（任意 scores.csv 组合的三级台阶 + abs 直考；基线复现验证 PASS）。

**2026-07-28 模型侧封板（§10.11 #1-4 + 快照系综全部关闭）**：
1. **多种子证伪**——**种子方差 ±6pt**（同配置 LoRA 单体 0.551/0.563/0.675）；多种子 = 稀释幸运成员（E6 0.701 < 0.755）。**历史所有单体 LoRA 数字按 ±6pt 噪声带判读。**
2. **全量微调关闭**——单体 0.596，容量非约束。
3. **405 盲评直督三臂证伪**——318 张独立照片是瓶颈（对数不构成多样性）；w1/w3 干扰单体，w0.3 温和锚自家考卷 0.488；405 库存下天花板已见。数据资产可复用：`audit/out/abs_pairs/`（train 20467 / test 924 / val 54；**abs test 对 = 最贴"用户的尺"的考卷**）。
4. **系综加权证伪**——val 单事件太薄不能做系综选择（search 选出 laion+l518，test 仅 0.684）。
5. 快照系综 0.744 无增益。

**剩余杠杆（全在数据侧）**：
**台阶② 现状（2026-07-28 用户容差重估）**：48 团 test 考卷已立（`golden_star/` 标星工作流，166 张）——**旧团顶一致率 0.732，团内旧标签作 ground truth 作废**；**正确口径 = 用户尺前二命中：系综 0.90（好片带 0.95），顶1 0.46 为副指标**；真模型错误仅 4/41（G005/G009/G013/G020）；不稳定集中茶博（0.44，差中选差），G026 极相似 tie 区实证（cos 0.978）。**下一步 = train 侧第二批标注（~100 团，加密极相似带）：扩考卷（n=41 薄）+ 修残余硬错误 + 提顶 1 决断力；48 团永久不入训**。工具链：`golden_star_export.py`（导出+剥星）/ `golden_star_readback.py`（读回）/ `golden_cluster_eval.py`（判决）。**纪律：组内标星 = 组内排序，禁止任何跨组数值比较；"差"的定义走 200 张横评事件级绝对尺。**
- **数据飞轮**：新外拍事件回流（跨场泛化根本欠账 = 仅 7 个训练事件；E4 已示目标应是个人尺）。
- **台阶① 可用性验收实验**（§10.11#7）：强/弱二分后人工复核率。

**判读纪律**：val 近 chance 时早停不可靠——固定 epoch 预算或多种子、报逐轮轨迹；**单体 LoRA 数字按 ±6pt 种子噪声带判读**；主指标组 = 段内 top-1 + seg-rho + Δ≥2 对级 + recall@12.5% + cos 分层（相似带）+ 三级台阶口径（m8_ensemble）；abs 涌现 n 小仅方向参考。

## 6. 工程守则（踩过的坑，别再踩）

- **命令统一在 `D:/Git/PhotoViewer`（仓根）跑**；探针/导出用 `Tools/.venv/Scripts/python.exe`（CPU），**训练用 `Tools/.venv-gpu/Scripts/python.exe`**（torch 2.11.0+cu128，4080 可用）；torch 运行带 `PYTHONUTF8=1`。
- **HF 门控 repo 401**：DINOv3 权重走 ModelScope 缓存（`~/.cache/modelscope/hub/models/facebook/dinov3-vit*16-pretrain-lvd1689m`，直接当 `--model-id`）；CLIP-L/14 同缓存（`AI-ModelScope/clip-vit-large-patch14`）；LAION 美学头在 `D:/PhotoDB/dataset/models/sac_logos_ava1-l14-linearMSE.pth`（MLP 768→1024→128→64→16→1，ReLU 注释、Dropout eval 恒等）。
- **渲染缓存正路**：`dotnet run --project Training/DatasetBuilder -- <文件夹或 --manifest> --dump-render D:/PhotoDB/dataset/render518`（与 DINO 预处理同一解码+缩放路径，PNG 无损，9418 张已全；**Python 复刻已证伪 cos 0.90，勿用**）。**DatasetBuilder 主流程与 dump-render 跑完均悬挂不退**（dispatcher 问题），GATE 打印后 kill 即可。
- **DB 只读访问用 `mode=ro` URI**；`photos.source_rel_path` + `event_label` → manifest 根目录解析源文件（旧批 NULL 事件 → `D:/PhotoDB/20240212`）。
- **文档纪律**：STATUS 每次会话结束重写（不追加）；EXECUTION-LOG 只追加；analysis-story 说人话（§10 = 训练阶段）；宪法修改带版本号（当前 v1.9）。
- **git**：按任务段合理阶段提交（用户已授权）；commit message 中文、阶段前缀。
- **铁律复核**：评估事件级切分 / 一切尺皆序 / EXIF 不进打分特征 / 闸门只卡真实差异（Δ≥2）/ 绝对尺只用团顶+孤立照 / 渲染同分布（余弦闸门）。

## 7. 文件地图

| 位置 | 内容 |
|---|---|
| `Training/STATUS.md` | 当前进度真源（每次会话结束重写） |
| `Training/EXECUTION-LOG.md` | 逐次实验台账（append-only） |
| `Training/docs/analysis-story.md` | 人读叙事版（**§10 = 训练阶段总叙事 + §10.11 后续优化预留表**） |
| `Training/docs/transfer-failure-analysis.md` | 失败矩阵详尽版（为什么失败 + 四轮证据） |
| `Training/plans/` | plan-3-0 宪法（v1.9）→ 3-1（M1）→ 3-2（M2+M3）→ 3-3（M4+M5）→ 3-4（M6+M7，后置） |
| `Training/audit/` | data_audit / cluster_mine / abs_set_sampler / m2_pool_builder / m2_offset_fit / m3_pair_gen / **abs_pair_gen（A1 盲评对）/ golden_cluster_sampler + golden_star_export + golden_star_readback + golden_cluster_eval（B1 金标准盲选链）** |
| `Training/train/` | **m5_lora.py（LoRA 训练+同口径评估，变体开关全：--seed/--w-abs/--abs-min-d/--full-ft）** · **m8_ensemble.py（系综+三级台阶考卷，已固化）** · m6_adapt_sim.py（事件内适配验证） · m7_extprobe.py（CLIP+LAION 外部先验探针） · render_cache.py（渲染闸门+路径解析） · m4_baseline.py（冻结特征基线，评估函数被复用） |
| `Training/DatasetBuilder/` | 入库管线 + **`--dump-render`（渲染缓存正路）** |
| `D:/PhotoDB/dataset/` | photos_dataset.db（9418 组四路特征+标签） · render518/（9418 PNG 缓存） · abs_set/m2_pool（盲评集+key+ratings） · **golden_star/（48 团 166 张盲选集，标星工作流，待标注）** · golden_clusters/（仅 key.csv 真值键） · models/（ONNX + LAION 头） |
| 关键数据资产 | `audit/out/m3_pairs/`（train 60180/val 1771/test 9542 + photos.csv） · `audit/out/abs_pairs/`（A1：train 20467/val 54/test 924） · `audit/out/m2_offset_clean/latent_scores.csv`（反泄漏干净潜分） · `audit/out/clusters/clusters.csv`（2569 团） · `train/out/m8_best/`（当前最优系综产物） |
