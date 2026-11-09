# 社交媒体多视角谣言检测 | Multi-view Social-media Rumor Detection

[中文说明](#中文说明) · [English](#english)

## 中文说明

### 项目简介
以社交媒体事件为样本，融合源帖的 TF-IDF 文本特征、传播结构特征和 Hawkes 风格时间特征，使用逻辑回归比较不同特征组合的分类表现。

本实现使用指数衰减激励统计量构建时间特征，并非拟合完整的 Hawkes 点过程。


### 方法流程
```text
事件标签 + 源帖文本 + 传播树
              ↓
文本特征 / 结构特征 / 时间特征
              ↓
分层划分训练集、验证集和测试集
              ↓
仅在训练集拟合 TF-IDF 与标准化器
              ↓
逻辑回归 → 对比实验 → Accuracy / Macro-F1
```

### 安装与运行
仓库整理时的离线验证使用 Windows / Python 3.12。先创建虚拟环境：
```bash
python -m venv .venv
```

Windows PowerShell：
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux：
```bash
source .venv/bin/activate
```

安装依赖并运行：
```bash
python -m pip install -r requirements.txt
python twitter_time_hawkes_rumor.py --dataset_dir /path/to/twitter15 --output_json results/metrics.json
```

将 `/path/to/twitter15` 替换为实际数据目录。请从原始来源获取数据并遵守其使用条件；仓库不分发原始推文、用户 ID 或传播树文件。

数据目录应包含：
```text
twitter15/
├── label.txt
├── source_tweets.txt
└── tree/
    └── <event_id>.txt
```

### 任务定义与实验设置
| 项目 | 默认设置或说明 |
| --- | --- |
| 分类任务 | 二分类：`non-rumor` 保持不变；`true`、`false`、`unverified` 合并为 `rumor`，并非真假二分类 |
| 四分类 | 使用 `--task_mode four_class` 保留四种标签 |
| 未证实标签 | `--no-include_unverified_in_rumor` 将其映射为 `non-rumor`，不是剔除样本 |
| 样本单位 | 每个事件一条样本 |
| 数据划分 | 分层划分，随机种子42；测试集占25%，验证集占剩余训练部分的15% |
| 预处理 | TF-IDF 和标准化器仅在训练集上拟合 |
| 指标 | Accuracy、Macro-F1 及分类报告 |
| 结果保存 | `--output_json` 保存配置、类别和各实验指标 |

对比文本、结构、时间及其组合。`no_time` 与 `text_tree` 相同，`no_tree` 与 `text_time` 相同，它们是别名而非独立基线。

结果表按验证集准确率排序。应使用验证集选择模型，避免根据反复查看的测试集结果调整参数。

### 测试与当前结果
```bash
python -m unittest discover -s tests -v
```

- 2项单元测试通过：时间特征维度一致性和带环结构的 BFS 终止。
- 使用80个虚构事件跑通了8项配置实验及 JSON 指标导出。
- 虚构数据测试只验证流程，不代表真实数据集性能；部分模型未预测某一类别，产生了精确率未定义警告。
- 本仓库尚未复现历史成绩，暂不声明真实数据集准确率。记录数据版本、配置和完整输出后，才能形成可核验的实验结果。

详见 [VERIFICATION.md](VERIFICATION.md)。

### 已知限制
- 传播结构当前以用户 ID 标识节点，同一用户的多次发帖可能被合并；解释树结构指标时需考虑这一表示方式。
- 时间特征取最早的 top-k 时间戳，结构特征使用完整传播树，因此不能据此宣称已经验证早期谣言检测效果。
- 推广结论前，应进一步检查重复文本、主题与时间重叠，以及数据划分的稳健性。

### 整理时的修改
- 修复空输入时间特征维度，使其与正常分支一致为31维。
- 增加未证实标签映射的显式关闭选项。
- 移除已弃用的分类器参数，增加 JSON 指标输出。
- 排除虚拟环境及第三方数据，保留直接依赖。

### 作者与许可
本项目为个人项目；应区分代码实现与外部研究、数据来源。第三方依赖遵循各自许可，本仓库目前未授予开源许可证。详见 [OWNERSHIP.md](OWNERSHIP.md)。

---

## English

### Overview
An event-level social-media classification project combining TF-IDF source-text features, propagation-structure features, and Hawkes-inspired temporal features. Logistic regression is used to compare different feature combinations.

The temporal representation uses exponentially decaying excitation statistics; it does not fit a complete Hawkes point process.

Author: 徐思哲 ([xizhexu845-glitch](https://github.com/xizhexu845-glitch)). This is an individual project.

### Method
```text
Event labels + source posts + propagation trees
                     ↓
Text / structural / temporal features
                     ↓
Stratified train / validation / test split
                     ↓
Fit TF-IDF and scalers on training data only
                     ↓
Logistic regression → comparisons → Accuracy / Macro-F1
```

### Installation and usage
Preparation-time offline checks used Windows / Python 3.12. Create a virtual environment:
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Install dependencies and run:
```bash
python -m pip install -r requirements.txt
python twitter_time_hawkes_rumor.py --dataset_dir /path/to/twitter15 --output_json results/metrics.json
```

Replace `/path/to/twitter15` with the actual dataset directory. Acquire data from its original source under its terms. Original posts, user IDs, and propagation-tree files are not redistributed here.

Expected layout:
```text
twitter15/
├── label.txt
├── source_tweets.txt
└── tree/
    └── <event_id>.txt
```

### Task definition and protocol
| Item | Default or explanation |
| --- | --- |
| Task | Binary: retain `non-rumor`; merge `true`, `false`, and `unverified` into `rumor`. This is not true-versus-false classification |
| Four-class mode | `--task_mode four_class` retains all four labels |
| Unverified labels | `--no-include_unverified_in_rumor` maps them to `non-rumor`; it does not exclude those samples |
| Sample unit | One row per event |
| Split | Stratified, seed 42; test fraction 25%, validation fraction 15% of the remaining training portion |
| Preprocessing | TF-IDF and scalers are fitted on training data only |
| Metrics | Accuracy, Macro-F1, and classification reports |
| Output | `--output_json` saves configuration, classes, and experiment metrics |

Compare text, structure, time, and combinations. `no_time` duplicates `text_tree`; `no_tree` duplicates `text_time`. These are aliases, not independent baselines.

The summary is sorted by validation accuracy. Select models using validation data, and avoid tuning parameters through repeated inspection of test results.

### Tests and current results
```bash
python -m unittest discover -s tests -v
```

- Two unit tests passed: temporal-feature dimensions and BFS termination on cyclic input.
- All eight configured experiments and JSON export completed on 80 invented events.
- Synthetic tests verify execution, not real-dataset performance. Some models did not predict one class and emitted undefined-precision warnings.
- Historical results have not been reproduced in this repository, so no real-dataset accuracy is claimed. Record the dataset version, configuration, and full output before reporting reproducible results.

See [VERIFICATION.md](VERIFICATION.md).

### Known limitations
- Propagation nodes are currently identified by user ID, so multiple posts from one user may be merged. Account for this representation when interpreting tree statistics.
- Temporal features use the earliest top-k timestamps, while structural features use the full tree. This is not a validated early-detection experiment.
- Check duplicate text, topic/time overlap, and split robustness before generalizing findings.

### Preparation changes
- Fixed the empty-input temporal-feature dimension to match the normal 31-dimensional output.
- Added an explicit false option for unverified-label mapping.
- Removed a deprecated classifier argument and added JSON metric output.
- Excluded virtual environments and third-party data; retained direct dependencies.

### Authorship and licensing
This is an individual project. Distinguish the implementation from external research and dataset sources. Dependencies retain their own licenses; no open-source license is currently granted for this repository. See [OWNERSHIP.md](OWNERSHIP.md).
