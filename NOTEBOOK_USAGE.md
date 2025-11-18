# Notebook 使用说明

## 概述

`complete_project.ipynb` 是一个完整的项目 notebook，它：
1. **调用训练脚本**（代码在 `train/` 文件夹中）
2. **进行结果分析和可视化**

## 工作流程

### 1. 训练模型（调用脚本）

Notebook 通过 `!python` 命令调用训练脚本：

- **Baseline 模型**: `!python train/run.py ...`
- **Hypothesis-Only 模型**: `!python train/train_hypothesis_only.py`
- **Debiased 模型**: `!python train/train_debiased.py`

### 2. 分析和可视化（在 notebook 中）

- 加载训练结果
- 进行错误分析
- 创建可视化图表
- 展示修复的例子

## 文件结构

```
fp-dataset-artifacts/
├── complete_project.ipynb          # 主 notebook
├── train/                          # 训练脚本（核心代码）
│   ├── run.py                      # Baseline 模型训练
│   ├── train_hypothesis_only.py    # Hypothesis-Only 模型训练
│   └── train_debiased.py           # Debiased 模型训练
├── analyze/                        # 分析脚本
│   ├── error_analysis.py
│   └── compare_models.py
└── outputs/evaluations/            # 训练结果
    ├── baseline_100k/
    ├── hypothesis_only_model/
    └── debiased_model/
```

## 使用方法

### 方法 1: 在 Jupyter Notebook 中运行

1. 打开 notebook：
   ```bash
   jupyter notebook complete_project.ipynb
   ```

2. 按顺序运行所有 cell

3. 训练脚本会自动执行并保存结果

### 方法 2: 在 Google Colab 中运行

1. 上传项目文件到 Colab

2. 确保路径正确（可能需要调整）

3. 运行所有 cell

### 方法 3: 单独运行脚本（不使用 notebook）

如果你想单独运行训练脚本：

```bash
# Baseline
python train/run.py --do_train --do_eval --task nli --dataset snli \
    --output_dir ./outputs/evaluations/baseline_100k/ \
    --max_train_samples 100000 --num_train_epochs 3

# Hypothesis-Only
python train/train_hypothesis_only.py

# Debiased
python train/train_debiased.py
```

然后使用 notebook 进行分析和可视化。

## 注意事项

### 关于 `!python` 命令

- 在 Jupyter notebook 中，`!` 是有效的 shell 命令前缀
- 编辑器可能会显示 linter 错误，但这是正常的
- 在 notebook 中运行时不会有问题

### 替代方案（如果需要）

如果你想避免 `!python` 命令，可以使用 `os.system()`：

```python
import os
os.system('python train/train_hypothesis_only.py')
```

或者使用 `subprocess`：

```python
import subprocess
subprocess.run(['python', 'train/train_hypothesis_only.py'])
```

### 路径问题

- 确保 notebook 在项目根目录运行
- 如果路径不对，调整 `BASELINE_DIR` 等变量

## 优势

这种设计的优势：

1. **代码模块化**: 训练代码在脚本中，可以独立运行和测试
2. **可重用性**: 脚本可以在命令行、notebook 或其他环境中使用
3. **清晰分离**: 训练代码和分析代码分离
4. **易于维护**: 修改训练逻辑只需修改脚本，不需要修改 notebook

## 故障排除

### 问题：找不到脚本

**解决方案**: 确保 notebook 在项目根目录运行

### 问题：训练失败

**解决方案**: 
1. 检查脚本是否可以直接运行
2. 查看错误信息
3. 确保所有依赖已安装

### 问题：结果文件不存在

**解决方案**: 
1. 检查训练是否成功完成
2. 查看输出目录是否正确
3. 手动运行脚本检查

## 下一步

训练完成后，notebook 会自动：
1. 加载结果
2. 进行对比分析
3. 创建可视化图表
4. 展示修复的例子

所有结果都会保存在 `outputs/evaluations/` 目录中。

