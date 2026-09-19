# PhotoViewer/Training — 照片美学排序模型训练仓库

PhotoViewer 的 **AI 训练独立仓库**（自主库拆分）：从产品仓库提取 DINOv3 特征 + CV 网格数据 → 构建独立训练数据集库 → 可行性探针 → 模型训练迭代。唯一目标是让模型理解摄影美学、对照片给出反映其水准的分数（准确性是唯一成功判据，暂不考虑产品化）。

仓库内容、构建运行方式、文档维护纪律见 [AGENTS.md](AGENTS.md)。

## 与主库的关系：两份仓库，各自维护

本仓物理上嵌套在主库工作区 `PhotoViewer/Training/` 下，但**是独立的 git 仓库**（主库 `.gitignore` 已忽略整个 `Training/` 目录）——克隆、拉取、提交都是两套：

- **拉取**：两个仓各自 `git pull`，互不影响，也不会互相携带更新；
- **提交**：改动落在哪边就在哪边提交——`Training/` 内的改动主库 `git status` 看不到；一次工作两边都有改动时（常见：`DatasetBuilder` 以 `ProjectReference` 复用 `PhotoViewer/Core`，提取算法演进常需双仓联动）**两个仓各提一笔**；
- 主库提交若影响训练侧（`model_id` / `cv_grid_spec` bump、schema 变化等），在提交信息里注明，Training 仓的同步改动单独提交；
- 操作前先 `git rev-parse --show-toplevel` 确认自己在哪个仓。