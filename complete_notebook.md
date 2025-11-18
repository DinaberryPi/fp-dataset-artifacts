# Complete Project Notebook - Dataset Artifacts Analysis

这是一个完整的 Jupyter notebook，包含项目的所有步骤。

---

## Cell 1: 安装依赖和导入库

```python
# 安装必要的包（如果在 Colab 或新环境中）
# !pip install transformers datasets torch tqdm evaluate accelerate

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
import torch
from tqdm.auto import tqdm

# 设置随机种子
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("✅ 所有库导入成功！")
```

---

## Cell 2: 添加项目路径和导入辅助函数

```python
# 添加项目路径
project_root = os.path.dirname(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入辅助函数
from preprocess.helpers import (
    prepare_dataset_nli,
    prepare_dataset_nli_hypothesis_only,
    compute_accuracy
)

print("✅ 辅助函数导入成功！")
```

---

## Cell 3: 加载数据集

```python
# 加载 SNLI 数据集
print("正在加载 SNLI 数据集...")
dataset = datasets.load_dataset('snli')

# 移除没有标签的例子
dataset = dataset.filter(lambda ex: ex['label'] != -1)

print(f"训练集大小: {len(dataset['train'])}")
print(f"验证集大小: {len(dataset['validation'])}")
print(f"测试集大小: {len(dataset['test'])}")

# 显示一个例子
print("\n示例数据:")
print(dataset['train'][0])
```

---

## Cell 4: 配置参数

```python
# 模型配置
MODEL_NAME = 'google/electra-small-discriminator'
MAX_TRAIN_SAMPLES = 100000  # 使用 100K 训练样本
MAX_EVAL_SAMPLES = None      # 使用全部验证集
NUM_EPOCHS = 3
BATCH_SIZE = 32
MAX_LENGTH = 128
LEARNING_RATE = 2e-5

# 输出目录
BASELINE_DIR = './outputs/evaluations/baseline_100k/'
HYPOTHESIS_ONLY_DIR = './outputs/evaluations/hypothesis_only_model/'
DEBIASED_DIR = './outputs/evaluations/debiased_model/'

# 创建输出目录
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(HYPOTHESIS_ONLY_DIR, exist_ok=True)
os.makedirs(DEBIASED_DIR, exist_ok=True)

print("✅ 配置完成！")
print(f"模型: {MODEL_NAME}")
print(f"训练样本数: {MAX_TRAIN_SAMPLES}")
print(f"训练轮数: {NUM_EPOCHS}")
```

---

## Cell 5: 准备数据集（Baseline）

```python
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 准备训练集（限制样本数）
train_dataset = dataset['train']
if MAX_TRAIN_SAMPLES:
    train_dataset = train_dataset.select(range(MAX_TRAIN_SAMPLES))

train_dataset = train_dataset.map(
    lambda ex: prepare_dataset_nli(ex, tokenizer, MAX_LENGTH),
    batched=True,
    num_proc=2,
    remove_columns=train_dataset.column_names
)

# 准备验证集
eval_dataset = dataset['validation']
if MAX_EVAL_SAMPLES:
    eval_dataset = eval_dataset.select(range(MAX_EVAL_SAMPLES))

eval_dataset = eval_dataset.map(
    lambda ex: prepare_dataset_nli(ex, tokenizer, MAX_LENGTH),
    batched=True,
    num_proc=2,
    remove_columns=eval_dataset.column_names
)

print(f"✅ 数据集准备完成！")
print(f"训练集: {len(train_dataset)} 样本")
print(f"验证集: {len(eval_dataset)} 样本")
```

---

## Cell 6: 训练 Baseline 模型

```python
print("=" * 80)
print("训练 Baseline 模型（Premise + Hypothesis）")
print("=" * 80)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# 训练参数
training_args = TrainingArguments(
    output_dir=BASELINE_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=f'{BASELINE_DIR}/logs',
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_accuracy,
)

# 训练
print("\n开始训练...")
trainer.train()

# 评估
print("\n评估模型...")
eval_results = trainer.evaluate()
print(f"\nBaseline 准确率: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")

# 保存模型
trainer.save_model()
print(f"\n✅ Baseline 模型已保存到: {BASELINE_DIR}")
```

---

## Cell 7: 生成 Baseline 预测

```python
# 生成预测
print("生成 Baseline 预测...")
predictions = trainer.predict(eval_dataset)

# 获取预测标签
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# 保存预测结果
predictions_data = []
for i, (true_label, pred_label) in enumerate(zip(true_labels, predicted_labels)):
    # 获取原始数据
    original_ex = dataset['validation'][i]
    predictions_data.append({
        'premise': original_ex['premise'],
        'hypothesis': original_ex['hypothesis'],
        'label': int(true_label),
        'predicted_label': int(pred_label)
    })

# 保存为 JSONL
with open(f'{BASELINE_DIR}/eval_predictions.jsonl', 'w', encoding='utf-8') as f:
    for item in predictions_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存指标
with open(f'{BASELINE_DIR}/eval_metrics.json', 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"✅ 预测已保存到: {BASELINE_DIR}/eval_predictions.jsonl")
print(f"✅ 指标已保存到: {BASELINE_DIR}/eval_metrics.json")
```

---

## Cell 8: 准备 Hypothesis-Only 数据集

```python
# 准备 hypothesis-only 训练集
print("准备 Hypothesis-Only 数据集（只使用 hypothesis，不使用 premise）...")

train_dataset_hyp = dataset['train']
if MAX_TRAIN_SAMPLES:
    train_dataset_hyp = train_dataset_hyp.select(range(MAX_TRAIN_SAMPLES))

train_dataset_hyp = train_dataset_hyp.map(
    lambda ex: prepare_dataset_nli_hypothesis_only(ex, tokenizer, MAX_LENGTH),
    batched=True,
    num_proc=2,
    remove_columns=train_dataset_hyp.column_names
)

# 准备验证集
eval_dataset_hyp = dataset['validation']
if MAX_EVAL_SAMPLES:
    eval_dataset_hyp = eval_dataset_hyp.select(range(MAX_EVAL_SAMPLES))

eval_dataset_hyp = eval_dataset_hyp.map(
    lambda ex: prepare_dataset_nli_hypothesis_only(ex, tokenizer, MAX_LENGTH),
    batched=True,
    num_proc=2,
    remove_columns=eval_dataset_hyp.column_names
)

print(f"✅ Hypothesis-Only 数据集准备完成！")
print(f"训练集: {len(train_dataset_hyp)} 样本")
print(f"验证集: {len(eval_dataset_hyp)} 样本")
```

---

## Cell 9: 训练 Hypothesis-Only 模型（Artifact Detector）

```python
print("=" * 80)
print("训练 Hypothesis-Only 模型（Artifact Detector）")
print("这个模型只看到 hypothesis，看不到 premise！")
print("如果准确率 > 33.33%（随机猜测），说明存在 artifacts！")
print("=" * 80)

# 加载新模型
hypothesis_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# 训练参数
training_args_hyp = TrainingArguments(
    output_dir=HYPOTHESIS_ONLY_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=f'{HYPOTHESIS_ONLY_DIR}/logs',
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
)

# 创建 Trainer
trainer_hyp = Trainer(
    model=hypothesis_model,
    args=training_args_hyp,
    train_dataset=train_dataset_hyp,
    eval_dataset=eval_dataset_hyp,
    compute_metrics=compute_accuracy,
)

# 训练
print("\n开始训练...")
trainer_hyp.train()

# 评估
print("\n评估模型...")
eval_results_hyp = trainer_hyp.evaluate()
hyp_accuracy = eval_results_hyp['eval_accuracy']
random_baseline = 1.0 / 3.0
above_random = hyp_accuracy - random_baseline

print(f"\nHypothesis-Only 准确率: {hyp_accuracy:.4f} ({hyp_accuracy*100:.2f}%)")
print(f"随机基线: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
print(f"高于随机: {above_random:.4f} ({above_random*100:.2f}%)")
print(f"\n{'✅ 检测到强 artifacts！' if above_random > 0.2 else '⚠️ 检测到弱 artifacts' if above_random > 0.1 else '❌ 未检测到明显 artifacts'}")

# 保存模型
trainer_hyp.save_model()
print(f"\n✅ Hypothesis-Only 模型已保存到: {HYPOTHESIS_ONLY_DIR}")
```

---

## Cell 10: 生成 Hypothesis-Only 预测

```python
# 生成预测
print("生成 Hypothesis-Only 预测...")
predictions_hyp = trainer_hyp.predict(eval_dataset_hyp)

# 获取预测标签
predicted_labels_hyp = np.argmax(predictions_hyp.predictions, axis=1)
true_labels_hyp = predictions_hyp.label_ids

# 保存预测结果
predictions_data_hyp = []
for i, (true_label, pred_label) in enumerate(zip(true_labels_hyp, predicted_labels_hyp)):
    original_ex = dataset['validation'][i]
    predictions_data_hyp.append({
        'premise': original_ex['premise'],
        'hypothesis': original_ex['hypothesis'],
        'label': int(true_label),
        'predicted_label': int(pred_label)
    })

# 保存为 JSONL
with open(f'{HYPOTHESIS_ONLY_DIR}/eval_predictions.jsonl', 'w', encoding='utf-8') as f:
    for item in predictions_data_hyp:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存指标
with open(f'{HYPOTHESIS_ONLY_DIR}/eval_metrics.json', 'w') as f:
    json.dump(eval_results_hyp, f, indent=2)

print(f"✅ 预测已保存到: {HYPOTHESIS_ONLY_DIR}/eval_predictions.jsonl")
```

---

## Cell 11: 定义 Debiased Trainer

```python
class DebiasedTrainer(Trainer):
    """
    自定义 Trainer，使用 hypothesis-only 模型来重新加权训练样本。
    对于 hypothesis-only 模型置信度高的样本，降低权重。
    """
    
    def __init__(self, *args, bias_model=None, bias_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_tokenizer = bias_tokenizer
        
        # 将 bias 模型移到相同设备
        if self.bias_model is not None:
            self.bias_model.to(self.args.device)
            self.bias_model.eval()  # 保持评估模式
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        基于 bias 模型的置信度计算加权损失。
        """
        labels = inputs.get("labels")
        
        # 获取主模型输出
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 获取 bias 模型预测（如果可用）
        weights = None
        if self.bias_model is not None:
            with torch.no_grad():
                # 为 bias 模型准备输入（只使用 hypothesis）
                # 注意：这里需要从原始输入中提取 hypothesis
                # 简化版本：假设 inputs 包含 input_ids
                bias_inputs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
                
                # 获取 bias 模型预测
                bias_outputs = self.bias_model(**bias_inputs)
                bias_logits = bias_outputs.logits
                
                # 计算置信度（最大 softmax 概率）
                bias_probs = torch.softmax(bias_logits, dim=-1)
                bias_confidence = torch.max(bias_probs, dim=-1)[0]
                
                # 计算权重：置信度越高，权重越低
                # weight = 1.0 / (1.0 + confidence)
                weights = 1.0 / (1.0 + bias_confidence)
        
        # 计算损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # 应用权重
        if weights is not None:
            loss = loss * weights.view(-1)
        
        # 返回平均损失
        return (loss.mean(), outputs) if return_outputs else loss.mean()

print("✅ DebiasedTrainer 类定义完成！")
```

---

## Cell 12: 训练 Debiased 模型

```python
print("=" * 80)
print("训练 Debiased 模型（使用 Hypothesis-Only 模型进行重加权）")
print("=" * 80)

# 加载新模型
debiased_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# 加载 hypothesis-only 模型作为 bias 模型
bias_model = AutoModelForSequenceClassification.from_pretrained(
    HYPOTHESIS_ONLY_DIR,
    num_labels=3
)

# 训练参数
training_args_deb = TrainingArguments(
    output_dir=DEBIASED_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir=f'{DEBIASED_DIR}/logs',
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
)

# 创建 Debiased Trainer
trainer_deb = DebiasedTrainer(
    model=debiased_model,
    args=training_args_deb,
    train_dataset=train_dataset,  # 使用完整的训练集（premise + hypothesis）
    eval_dataset=eval_dataset,
    compute_metrics=compute_accuracy,
    bias_model=bias_model,
    bias_tokenizer=tokenizer,
)

# 训练
print("\n开始训练...")
trainer_deb.train()

# 评估
print("\n评估模型...")
eval_results_deb = trainer_deb.evaluate()
print(f"\nDebiased 准确率: {eval_results_deb['eval_accuracy']:.4f} ({eval_results_deb['eval_accuracy']*100:.2f}%)")

# 保存模型
trainer_deb.save_model()
print(f"\n✅ Debiased 模型已保存到: {DEBIASED_DIR}")
```

---

## Cell 13: 生成 Debiased 预测

```python
# 生成预测
print("生成 Debiased 预测...")
predictions_deb = trainer_deb.predict(eval_dataset)

# 获取预测标签
predicted_labels_deb = np.argmax(predictions_deb.predictions, axis=1)
true_labels_deb = predictions_deb.label_ids

# 保存预测结果
predictions_data_deb = []
for i, (true_label, pred_label) in enumerate(zip(true_labels_deb, predicted_labels_deb)):
    original_ex = dataset['validation'][i]
    predictions_data_deb.append({
        'premise': original_ex['premise'],
        'hypothesis': original_ex['hypothesis'],
        'label': int(true_label),
        'predicted_label': int(pred_label)
    })

# 保存为 JSONL
with open(f'{DEBIASED_DIR}/eval_predictions.jsonl', 'w', encoding='utf-8') as f:
    for item in predictions_data_deb:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存指标
with open(f'{DEBIASED_DIR}/eval_metrics.json', 'w') as f:
    json.dump(eval_results_deb, f, indent=2)

print(f"✅ 预测已保存到: {DEBIASED_DIR}/eval_predictions.jsonl")
```

---

## Cell 14: 结果汇总

```python
# 加载所有指标
with open(f'{BASELINE_DIR}/eval_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

with open(f'{HYPOTHESIS_ONLY_DIR}/eval_metrics.json', 'r') as f:
    hyp_metrics = json.load(f)

with open(f'{DEBIASED_DIR}/eval_metrics.json', 'r') as f:
    debiased_metrics = json.load(f)

# 计算统计
random_baseline = 1.0 / 3.0
baseline_acc = baseline_metrics['eval_accuracy']
hyp_acc = hyp_metrics['eval_accuracy']
debiased_acc = debiased_metrics['eval_accuracy']

print("=" * 80)
print("结果汇总")
print("=" * 80)
print(f"\n随机基线:        {random_baseline:.4f} ({random_baseline*100:.2f}%)")
print(f"Hypothesis-Only: {hyp_acc:.4f} ({hyp_acc*100:.2f}%) [高于随机: +{(hyp_acc-random_baseline)*100:.2f}%]")
print(f"Baseline:        {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"Debiased:        {debiased_acc:.4f} ({debiased_acc*100:.2f}%) [变化: {(debiased_acc-baseline_acc)*100:+.2f}%]")

print("\n" + "=" * 80)
print("关键发现:")
print("=" * 80)
print(f"1. Hypothesis-Only 模型达到 {hyp_acc*100:.2f}%，证明存在强 artifacts！")
print(f"2. Debiasing 后准确率变化: {(debiased_acc-baseline_acc)*100:+.2f}%")
print(f"3. {'✅ Debiasing 保持了性能' if abs(debiased_acc - baseline_acc) < 0.01 else '⚠️ Debiasing 影响了性能'}")
```

---

## Cell 15: 错误分析 - Baseline 模型

```python
# 加载 Baseline 预测
baseline_predictions = []
with open(f'{BASELINE_DIR}/eval_predictions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        baseline_predictions.append(json.loads(line))

print("=" * 80)
print("Baseline 模型错误分析")
print("=" * 80)

# 标签名称
label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# 计算总体准确率
correct = sum(1 for p in baseline_predictions if p['label'] == p['predicted_label'])
total = len(baseline_predictions)
accuracy = correct / total
print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"正确: {correct}/{total}")
print(f"错误: {total - correct}/{total} ({(total-correct)/total:.1%})")

# 标签分布
print("\n标签分布:")
true_labels = Counter(p['label'] for p in baseline_predictions)
for label, count in sorted(true_labels.items()):
    print(f"  {label_names[label]}: {count} ({count/total:.1%})")

# 混淆矩阵
print("\n混淆矩阵 (行=真实标签, 列=预测标签):")
confusion = defaultdict(lambda: defaultdict(int))
for p in baseline_predictions:
    confusion[p['label']][p['predicted_label']] += 1

print(f"{'':20} {'Entail':>10} {'Neutral':>10} {'Contrad':>10}")
for true_label in [0, 1, 2]:
    row = f"{label_names[true_label]:20}"
    for pred_label in [0, 1, 2]:
        count = confusion[true_label][pred_label]
        row += f"{count:>10}"
    print(row)

# 每类准确率
print("\n每类准确率:")
for label in [0, 1, 2]:
    total_for_label = true_labels[label]
    correct_for_label = confusion[label][label]
    acc = correct_for_label / total_for_label if total_for_label > 0 else 0
    print(f"  {label_names[label]:15}: {acc:.2%} ({correct_for_label}/{total_for_label})")
```

---

## Cell 16: 错误分析 - 否定词分析

```python
# 分析否定词
negation_words = ['no', 'not', 'never', 'nobody', 'nothing', 'nowhere', 'neither', 'none', "n't"]

def has_negation(text):
    text_lower = text.lower()
    return any(neg in text_lower for neg in negation_words)

# 找出包含否定词的假设
hyp_with_negation = [p for p in baseline_predictions if has_negation(p['hypothesis'])]
hyp_without_negation = [p for p in baseline_predictions if not has_negation(p['hypothesis'])]

print("=" * 80)
print("否定词分析")
print("=" * 80)
print(f"\n包含否定词的假设: {len(hyp_with_negation)} ({len(hyp_with_negation)/len(baseline_predictions):.1%})")
print(f"不包含否定词的假设: {len(hyp_without_negation)} ({len(hyp_without_negation)/len(baseline_predictions):.1%})")

if hyp_with_negation:
    # 真实标签分布
    neg_true_labels = Counter(p['label'] for p in hyp_with_negation)
    print(f"\n包含否定词的假设 - 真实标签分布:")
    for label, count in sorted(neg_true_labels.items()):
        print(f"  {label_names[label]}: {count} ({count/len(hyp_with_negation):.1%})")
    
    # 预测标签分布
    neg_pred_labels = Counter(p['predicted_label'] for p in hyp_with_negation)
    print(f"\n包含否定词的假设 - 预测标签分布:")
    for label, count in sorted(neg_pred_labels.items()):
        print(f"  {label_names[label]}: {count} ({count/len(hyp_with_negation):.1%})")
    
    # 准确率
    neg_correct = sum(1 for p in hyp_with_negation if p['label'] == p['predicted_label'])
    neg_acc = neg_correct / len(hyp_with_negation)
    print(f"\n包含否定词的假设 - 准确率: {neg_acc:.2%}")
```

---

## Cell 17: 模型对比 - Baseline vs Debiased

```python
# 加载 Debiased 预测
debiased_predictions = []
with open(f'{DEBIASED_DIR}/eval_predictions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        debiased_predictions.append(json.loads(line))

print("=" * 80)
print("Baseline vs Debiased 对比")
print("=" * 80)

# 确保长度相同
min_len = min(len(baseline_predictions), len(debiased_predictions))
baseline_preds = baseline_predictions[:min_len]
debiased_preds = debiased_predictions[:min_len]

# 总体准确率
baseline_correct = sum(1 for p in baseline_preds if p['label'] == p['predicted_label'])
debiased_correct = sum(1 for p in debiased_preds if p['label'] == p['predicted_label'])

baseline_acc = baseline_correct / len(baseline_preds)
debiased_acc = debiased_correct / len(debiased_preds)

print(f"\n总体准确率:")
print(f"  Baseline: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"  Debiased: {debiased_acc:.4f} ({debiased_acc*100:.2f}%)")
print(f"  变化:     {(debiased_acc-baseline_acc)*100:+.2f}%")

# 每类准确率
print(f"\n每类准确率:")
for label in [0, 1, 2]:
    baseline_class = [p for p in baseline_preds if p['label'] == label]
    debiased_class = [p for p in debiased_preds if p['label'] == label]
    
    baseline_class_acc = sum(1 for p in baseline_class if p['predicted_label'] == label) / len(baseline_class)
    debiased_class_acc = sum(1 for p in debiased_class if p['predicted_label'] == label) / len(debiased_class)
    
    change = debiased_class_acc - baseline_class_acc
    print(f"  {label_names[label]:15}: Baseline={baseline_class_acc:.2%}, Debiased={debiased_class_acc:.2%}, Change={change:+.2%}")

# 预测变化
changes = []
for i, (base, deb) in enumerate(zip(baseline_preds, debiased_preds)):
    if base['predicted_label'] != deb['predicted_label']:
        changes.append({
            'index': i,
            'premise': base['premise'],
            'hypothesis': base['hypothesis'],
            'true_label': base['label'],
            'baseline_pred': base['predicted_label'],
            'debiased_pred': deb['predicted_label'],
        })

print(f"\n预测变化:")
print(f"  总变化数: {len(changes)} ({len(changes)/len(baseline_preds):.1%})")

# 分类变化
baseline_wrong_debiased_right = [c for c in changes if c['baseline_pred'] != c['true_label'] and c['debiased_pred'] == c['true_label']]
baseline_right_debiased_wrong = [c for c in changes if c['baseline_pred'] == c['true_label'] and c['debiased_pred'] != c['true_label']]

print(f"  Baseline 错 -> Debiased 对 (修复): {len(baseline_wrong_debiased_right)}")
print(f"  Baseline 对 -> Debiased 错 (破坏): {len(baseline_right_debiased_wrong)}")
print(f"  净改进: {len(baseline_wrong_debiased_right) - len(baseline_right_debiased_wrong):+d}")
```

---

## Cell 18: 可视化 - 结果对比

```python
# 创建结果对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 总体准确率对比
models = ['Random', 'Hypothesis-\nOnly', 'Baseline', 'Debiased']
accuracies = [random_baseline, hyp_acc, baseline_acc, debiased_acc]
colors = ['gray', 'orange', 'blue', 'green']

axes[0].bar(models, accuracies, color=colors, alpha=0.7)
axes[0].axhline(y=random_baseline, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Overall Model Performance')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
for i, (model, acc) in enumerate(zip(models, accuracies)):
    axes[0].text(i, acc + 0.02, f'{acc:.2%}', ha='center', va='bottom')

# 图2: 每类准确率对比
classes = ['Entailment', 'Neutral', 'Contradiction']
baseline_class_accs = []
debiased_class_accs = []

for label in [0, 1, 2]:
    baseline_class = [p for p in baseline_preds if p['label'] == label]
    debiased_class = [p for p in debiased_preds if p['label'] == label]
    
    baseline_class_acc = sum(1 for p in baseline_class if p['predicted_label'] == label) / len(baseline_class)
    debiased_class_acc = sum(1 for p in debiased_class if p['predicted_label'] == label) / len(debiased_class)
    
    baseline_class_accs.append(baseline_class_acc)
    debiased_class_accs.append(debiased_class_acc)

x = np.arange(len(classes))
width = 0.35
axes[1].bar(x - width/2, baseline_class_accs, width, label='Baseline', alpha=0.7, color='blue')
axes[1].bar(x + width/2, debiased_class_accs, width, label='Debiased', alpha=0.7, color='green')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Per-Class Accuracy Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].legend()
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/evaluations/results_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存到: ./outputs/evaluations/results_comparison.png")
plt.show()
```

---

## Cell 19: 可视化 - 混淆矩阵

```python
# 创建混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline 混淆矩阵
baseline_confusion = np.zeros((3, 3))
for p in baseline_preds:
    baseline_confusion[p['label']][p['predicted_label']] += 1

# 归一化
baseline_confusion_norm = baseline_confusion / baseline_confusion.sum(axis=1, keepdims=True)

sns.heatmap(baseline_confusion_norm, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Entail', 'Neutral', 'Contrad'],
            yticklabels=['Entail', 'Neutral', 'Contrad'],
            ax=axes[0], cbar_kws={'label': 'Proportion'})
axes[0].set_title('Baseline Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Debiased 混淆矩阵
debiased_confusion = np.zeros((3, 3))
for p in debiased_preds:
    debiased_confusion[p['label']][p['predicted_label']] += 1

# 归一化
debiased_confusion_norm = debiased_confusion / debiased_confusion.sum(axis=1, keepdims=True)

sns.heatmap(debiased_confusion_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Entail', 'Neutral', 'Contrad'],
            yticklabels=['Entail', 'Neutral', 'Contrad'],
            ax=axes[1], cbar_kws={'label': 'Proportion'})
axes[1].set_title('Debiased Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('./outputs/evaluations/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ 混淆矩阵已保存到: ./outputs/evaluations/confusion_matrices.png")
plt.show()
```

---

## Cell 20: 展示修复的例子

```python
# 展示一些修复的例子
print("=" * 80)
print("Debiasing 修复的例子")
print("=" * 80)

fixes = baseline_wrong_debiased_right[:5]  # 显示前5个

for i, fix in enumerate(fixes, 1):
    print(f"\n修复例子 {i}:")
    print(f"  Premise: {fix['premise']}")
    print(f"  Hypothesis: {fix['hypothesis']}")
    print(f"  真实标签: {label_names[fix['true_label']]}")
    print(f"  Baseline 预测: {label_names[fix['baseline_pred']]} ❌")
    print(f"  Debiased 预测: {label_names[fix['debiased_pred']]} ✅")
    print("-" * 80)
```

---

## Cell 21: 总结和下一步

```python
print("=" * 80)
print("项目总结")
print("=" * 80)

print("\n✅ 已完成:")
print("  1. Baseline 模型训练和评估")
print("  2. Hypothesis-Only 模型训练（Artifact 检测）")
print("  3. Debiased 模型训练（使用重加权方法）")
print("  4. 错误分析和模型对比")
print("  5. 可视化结果")

print("\n📊 关键结果:")
print(f"  - Hypothesis-Only: {hyp_acc*100:.2f}% (高于随机 +{(hyp_acc-random_baseline)*100:.2f}%)")
print(f"  - Baseline: {baseline_acc*100:.2f}%")
print(f"  - Debiased: {debiased_acc*100:.2f}% (变化: {(debiased_acc-baseline_acc)*100:+.2f}%)")

print("\n📝 下一步:")
print("  1. 分析结果并撰写论文")
print("  2. 创建更多可视化（如果需要）")
print("  3. 深入分析特定错误类型")
print("  4. 准备论文的表格和图表")

print("\n📁 输出文件:")
print(f"  - Baseline 预测: {BASELINE_DIR}/eval_predictions.jsonl")
print(f"  - Hypothesis-Only 预测: {HYPOTHESIS_ONLY_DIR}/eval_predictions.jsonl")
print(f"  - Debiased 预测: {DEBIASED_DIR}/eval_predictions.jsonl")
print(f"  - 结果对比图: ./outputs/evaluations/results_comparison.png")
print(f"  - 混淆矩阵: ./outputs/evaluations/confusion_matrices.png")

print("\n" + "=" * 80)
print("项目完成！🎉")
print("=" * 80)
```

---

## 使用说明

1. **在 Jupyter Notebook 中使用:**
   - 将每个 cell 复制到 Jupyter notebook 中
   - 按顺序运行所有 cell
   - 确保项目结构正确

2. **在 Google Colab 中使用:**
   - 上传项目文件到 Colab
   - 修改路径设置
   - 运行所有 cell

3. **注意事项:**
   - 训练可能需要较长时间（特别是 Debiased 模型）
   - 确保有足够的 GPU 内存
   - 可以调整 `MAX_TRAIN_SAMPLES` 来减少训练时间

---

*Notebook 创建日期: 2024年11月*

