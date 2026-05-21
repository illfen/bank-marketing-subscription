# 银行电话营销客户认购预测

本项目基于 UCI Bank Marketing 数据集，预测银行电话营销客户是否会认购定期存款。仓库只保留实验代码、测试文件和已生成的实验结果，不包含课程报告正文、PDF、LaTeX 源文件或签名图片。

## 项目结构

- `src/bank_marketing/`：数据读取、预处理、模型构建、评价指标和绘图函数。
- `scripts/run_experiments.py`：完整实验运行脚本。
- `tests/`：预处理和评价逻辑的单元测试。
- `outputs/`：已生成的指标文件和图像结果。
- `requirements.txt`：Python 依赖列表。

## 环境配置

使用 conda 创建环境：

```bash
conda create -y -p ./.conda python=3.10 pip
conda run -p ./.conda python -s -m pip install -r requirements.txt
```

如果已经激活自己的 Python 环境，也可以直接安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 运行测试

```bash
conda run -p ./.conda python -s -m pytest tests -q
```

或：

```bash
python -m pytest tests -q
```

## 运行实验

```bash
conda run -p ./.conda python -s scripts/run_experiments.py
```

或：

```bash
python scripts/run_experiments.py
```

脚本会自动下载 UCI Bank Marketing 数据集到 `data/raw/`，完成模型训练、阈值选择、指标计算和图像生成，并将结果保存到 `outputs/`。

## 输出结果

- `outputs/metrics.csv`：各模型的 Accuracy、Precision、Recall、F1、ROC-AUC、Average Precision 等指标。
- `outputs/dataset_profile.json`：数据集基本统计信息。
- `outputs/duration_leakage_check.csv`：`duration` 字段的数据泄漏对照实验结果。
- `outputs/metric_comparison.png`：模型指标对比图。
- `outputs/roc_curves.png`：ROC 曲线图。
- `outputs/best_confusion_matrix.png`：最优模型混淆矩阵。

## 实验说明

主实验默认排除 `duration` 字段。该字段表示电话通话持续时间，只有电话结束后才能知道；如果用于营销前预测，会造成数据泄漏，导致离线指标虚高。
