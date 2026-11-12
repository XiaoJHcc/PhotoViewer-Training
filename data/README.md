# 数据契约 Data Contract

> 描述训练数据集库的物理位置、schema、与产品库的对齐关系、版本化规则。字段以 [DatasetDatabase.cs](../DatasetBuilder/DatasetDatabase.cs) 的实际建表语句为准,本文档随其同步更新,不重复维护第二份 schema 定义。

## 位置

数据集库(SQLite 文件)与原始照片本体都在**仓外**,不入库、不受仓库清理影响,便携:

- 默认落盘位置:`D:\PhotoDB\dataset\photos_dataset.db`。
- 每次 `DatasetBuilder` 运行的清单(见 [DatasetBuilder/manifest.sample.json](../DatasetBuilder/manifest.sample.json))可以指定任意 `dbPath`——按拍摄批次拆分成多个独立库,或合并写入同一个库,由使用者决定。
- 覆盖率报告落在 `<dbPath>.coverage.md`,与库文件同目录。

## 与产品 `photos.db` 的对齐关系

数据集库由 [DatasetDatabase](../DatasetBuilder/DatasetDatabase.cs) 门面管理,是与产品 `PhotoDatabase`(见 [PhotoViewer/Core/Database/AGENTS.md](../../PhotoViewer/Core/Database/AGENTS.md))**完全解耦的独立库**,但刻意让核心表名/列名与产品对齐,方便 Python 侧沿用同一套读法。区别:

1. **路径由清单指定**,不落 AppData——入库后不再触碰原始照片文件。
2. `photos` 表在产品列之外**加了训练专用列**:`is_retouched`(精修命中)/ `source_rel_path`(相对路径)/ `event_label`(事件标签)/ `subject_label`(题材标签)/ `formats`(该指纹合一的格式集合)。
3. `photo_features` 每个指纹可以存**两行**——原片 CLS(`model_id`)+ 增强 CLS(`model_id` 带 `+clhe.. ycc..` 后缀),支撑探针做"仅原片 / 仅增强 / 多视图"对比,产品库只有一行。
4. **迁移只做加列(additive)**——数据集是精选产物,绝不会像产品库那样在 schema 不匹配时删库重建;新增列走 `ALTER TABLE ADD COLUMN`,历史数据保留。
5. `dataset_meta` 键值表记录模型 id / 增强参数 / CV 版本 / 预处理分辨率等,保证同一批数据可复现追溯。

## 三表 schema

### `photos`(身份 + EXIF/rating + CV grid + 训练来源标签)

| 列 | 类型 | 说明 |
|---|---|---|
| `fingerprint` | TEXT PK | 同次曝光的 RAW/HIF/JPG 共享同一指纹 |
| `filename_noext` / `capture_time` / `capture_subsec` | TEXT | 指纹身份三要素 |
| `rating` | INTEGER | 星级；只写实际选用文件（指纹组代表件）自身内嵌 XMP 的 `Rated` 1–5（口径 = 产品 `XmpRatingReader`），伴侣文件与 sidecar 不读、不取 max；0★、缺失、`Invalid`、`Conflict`、`Rejected` 一律不写。COALESCE 写入,不会被空值覆盖抹掉；0★ 由批次语义回填 |
| `focal_length` / `aperture` / `shutter_speed` / `crop_factor` | REAL | EXIF 快照,原始值(等效焦距/光圈不入库,训练脚本自行换算) |
| `cv_grid` | BLOB | CV 网格 7 标量(含 `block_contrast`),编码方式见 `CvGridResult.Encode()` |
| `cv_grid_spec` | TEXT | CV 算法版本串,与当前 `CvGridResult.CurrentVersion` 不匹配即视为该项缺失需要重算 |
| `cv_computed_at` / `cv_image_width` / `cv_image_height` | TEXT/INTEGER | CV 计算时间戳与原图分辨率(diagonal 等派生量靠这俩推) |
| `is_retouched` | INTEGER | 精修命中(1/0);清单未提供精修清单时全 NULL(未知,不占位) |
| `source_rel_path` / `event_label` / `subject_label` / `formats` | TEXT | 来自入库清单的来源标签,每次入库覆盖刷新 |
| `updated_at` | TEXT | 最近写入时间(UTC ISO) |

### `photo_features`(CLS 纵表,`(fingerprint, model_id)` 联合主键)

| 列 | 类型 | 说明 |
|---|---|---|
| `fingerprint` | TEXT | 外键 → `photos.fingerprint`,级联删除 |
| `model_id` | TEXT | 原片 CLS 用 `DinoModelResources.ModelId`;增强 CLS 用其后缀变体(`+clheX.Xycc X.X`) |
| `cls_vector` | BLOB | 384 维 float32,L2 归一化,`float[]` 原始字节序编码 |
| `computed_at` | TEXT | 计算时间(UTC ISO) |

### `photo_patches`(patch token 纵表,同主键结构)

| 列 | 类型 | 说明 |
|---|---|---|
| `fingerprint` / `model_id` | TEXT | 同上;patch 只存原片一行,不存增强路 |
| `patch_tokens` | BLOB | 1024×384 float32,token-major 排布(`PatchTokenCount × FeatureDim`) |
| `computed_at` | TEXT | 计算时间 |

### `dataset_meta`(键值契约表)

`(key, value, updated_at)`,`INSERT ... ON CONFLICT DO UPDATE` 覆盖式写入。当前写入的键(见 [Program.cs](../DatasetBuilder/Program.cs) `RunPipelineAsync`):`dino_model_id` / `enhanced_model_id` / `clip_factor` / `saturation_scale` / `color_model` / `cv_spec` / `dino_input_size` / `enhance_resolution`。

## 版本化 = 改算法即整库失效

`model_id`(DINO)与 `cv_grid_spec`(CV)是**唯一**的缓存有效性判据:

- 改 DINO 预处理/模型权重 → bump `DinoModelResources.ModelId` → 旧行的 `model_id` 不匹配,`EvaluateMissingAsync` 判定缺失,重新提取写入新行(旧行不删,历史可追溯)。
- 改增强算法参数(对比度裁剪、饱和度缩放、色彩模型)→ 增强 `model_id` 后缀自动随 `ImageEnhancer.ClipFactor` / `SaturationScale` 变化 → 同样触发按需重算,不会用旧算法产物静默污染新实验。
- 改 CV 网格算法 → bump `CvGridResult.CurrentVersion` → `cv_grid_spec` 不匹配同理触发重算。

**齐备度(GATE)判定分母是可解码组数**,不是全部指纹组——仅 RAW 且当前无解码器的组只写身份/EXIF/来源标签,不提特征,属合法缺失,不计入失败(见 [CoverageReport.cs](../DatasetBuilder/CoverageReport.cs))。

## 消费方

Python 探针([../probes/](../probes/))只通过上述表结构契约读取数据,不依赖任何 C# 代码;只要表名/列名/版本化规则不破坏性变更,探针脚本不需要跟着改。

## 修复技术侧缓存（2026-10-14）

`DatasetBuilder --dump-technical` 的独立输出位于仓外 `D:/PhotoDB/dataset/technical-display-plane-v3-*`，不修改上述 SQLite schema 或存量数据。来源为固定 `Fingerprint/Path` JSON；共享算法仍来自产品 Core，训练侧不复刻 CV/DINO。

- `identity.json`：`technical-cache-v1`、解码版本、带 decode 后缀的 cv_spec/model_id、来源清单 SHA256、产品/提取器程序集 SHA256、可选 ONNX SHA256、标量名/网格/输入尺寸与输出选项。身份不符拒绝续跑。
- `<fingerprint>.cv.bin`：小端 float32 `[7,32,32]`、scalar-major；可选 `.patch.bin` 为 `[1024,384]` token-major，`.cls.bin` 为归一化 CLS，`.png` 为共享 `RenderInputImage` 产物。
- `<fingerprint>.json`：实际显示朝向宽高、原片路径与 SHA256、每项输出 SHA256；缺失读数保持 NaN。原片指纹通过产品 PhotoFingerprint 重算核验，哈希在提取前后复核；输出完成后才落单图记录。
- `complete.json`：请求/完成数量与 identity SHA256，只有全部请求成功才写入；输入或输出漂移直接报错，不静默使用残缺缓存。
- 布局不等于严格坐标重合：CV 格中心为 `(index+0.5)/32`，DINO 518 输入的 patch 中心为 `(index+0.5)*16/518`，patch 只覆盖前512像素。做像素级融合前须明确这一几何差异；本轮聚合探针不使用旧 patch。

核验入口与职责见 [../AGENTS.md](../AGENTS.md)；实测差异及效力边界见 [../docs/KNOWLEDGE.md](../docs/KNOWLEDGE.md) §4。

## 修复 patch 网格缓存（2026-11-11）

`train/cache_patch_tokens.py` 的仓外输出 `D:/PhotoDB/dataset/patches-display-plane-v3-dinov3s-*`，只消费 `render518-display-plane-v3` 修复 PNG，不修改数据集库。现有目录覆盖 v2 train+val 18331 张，test 未提取。

- `patches.f16.npy`：`[照片, 1024, 384]` float16，token 为 32×32 行优先网格，取 `last_hidden_state` 末 1024 个 token（跳过 CLS 与 4 个 register），与产品 ONNX `patch_tokens` 同义；fp32 前向。
- `fingerprints.json`：行序 = `corrected_features.npz` 指纹顺序中属于所选 split 的子集。
- `identity.json`：解码版本、split、权重/照片清单/修复特征/代码 SHA256、产品 ONNX 一致性闸统计；身份不符拒绝续跑。
- `progress.json`：逐图渲染 SHA256，提取时逐一对照修复特征清单；`complete.json`：全覆盖后的 patch 文件 SHA256，消费方须校验后再用。
- 几何口径同上节：patch 只覆盖 518 输入的前 512 像素。

## 保持宽高比预览缓存（2026-11-11）

`--dump-technical ... --preview-long-edge N [--no-cv]` 的仓外输出 `D:/PhotoDB/dataset/preview1024-display-plane-v3-train-20261111`（v2 train 12894 张，长边 1024，约 18GB）。像素路径 = 修复 `PhotoDecode`（显示朝向）→ 产品 `ColorEngine.ToSrgb8` → `SkiaImageCodec.ScaleToLongEdge`（不放大、不裁切）→ PNG。
- `identity.json`：`preview_long_edge` 与 `cv_spec`（`--no-cv` 时为 null）入身份，其余同技术侧缓存；来源清单不同必须换目录。
- `<fingerprint>.preview.png` 与单图记录中的 `preview_sha256`；`--no-cv` 时不写 `.cv.bin`，`cv_sha256` 为 null。
- 用途：需要正确宽高比的外部模型（如零样本美学/画质）；不得替代 518 方形 DINO 输入。
