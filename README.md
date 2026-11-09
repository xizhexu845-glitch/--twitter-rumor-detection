# Multi-view Social-media Rumor Detection
TF-IDF text, propagation structure and Hawkes-inspired temporal features with logistic regression.

## 项目概述
以社交媒体事件为样本，融合源帖文本、传播结构和时间特征，比较不同特征组合在谣言识别任务中的表现。该实现使用 Hawkes 风格的指数衰减激励统计量，并非拟合完整 Hawkes 点过程。

## Setup and run
Use Python 3.11:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python twitter_time_hawkes_rumor.py --dataset_dir /path/to/twitter15 --output_json results/metrics.json
```
Acquire the dataset from its original source under its terms. No original tweets, user IDs or dataset trees are redistributed here.
Expected layout: `label.txt`, `source_tweets.txt`, and `tree/<event_id>.txt`.

## Experiment protocol
- Default: binary classification; `non-rumor` remains non-rumor, while `true`, `false`, and `unverified` are merged into rumor. This is not a true-versus-false classifier.
- `--task_mode four_class` retains the four labels.
- `--no-include_unverified_in_rumor` maps unverified to non-rumor (it does NOT exclude these samples).
- One row per event; stratified train/dev/test split, random seed 42. Test fraction 25%; validation is 15% of the remaining training portion.
- TF-IDF and scalers are fitted on training data only.
- Compare text-only, structure-only, time-only and combined features. `no_time` duplicates `text_tree`; `no_tree` duplicates `text_time` and are aliases, not independent baselines.
- Sort the summary by validation accuracy; use validation, not test performance, for model choice. Avoid tuning parameters based on repeatedly viewed test metrics.

## Results and caveats
No historical accuracy is claimed by this repository until a reproducible run is recorded with its data version, configuration and output. Synthetic tests only check program behavior.
Tree nodes currently use user IDs, so repeated posts by one user may be merged; assess this representation before interpreting propagation-tree statistics.
Temporal features use earliest top-k timestamps while structural features use the full tree. This is not a validated early-detection experiment.
Check duplicate content, topic/time overlap and split robustness before generalizing results.

## Changes during preparation
- Fixed empty temporal-feature dimension (31).
- Added an explicit false option for unverified-label mapping.
- Removed deprecated classifier argument; added machine-readable metric output.
- Removed virtual environment and third-party data; retained minimal direct dependencies.

## Ownership
Personal project by 徐思哲 (xizhexu845-glitch). See OWNERSHIP.md. The implementation and data must be distinguished from any referenced research; original dataset files are not included.

## Offline checks
```bash
python -m unittest discover -s tests -v
```
