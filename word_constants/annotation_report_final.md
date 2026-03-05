# 性别偏见词典标注信度分析报告 (Final)

## 1. 任务概述

- **样本量**: 45 个词语
- **数据源**: `labeled_results1.csv` vs `labeled_results2.csv`

## 2. 信度指标 (Reliability)

- **总一致率 (Percentage Agreement)**: 88.89%
- **Cohen's Kappa 系数**: 0.864
- **解读**: Kappa = 0.864，属于 **极高一致**。

### 维度详细报告 (以 Label 1 为基准)

| 维度                  | Precision | Recall | F1-Score | Support |
| ------------------- | --------- | ------ | -------- | ------- |
| Agentic             | 0.818     | 1.000  | 0.900    | 9.0     |
| Communal            | 1.000     | 1.000  | 1.000    | 4.0     |
| Effort              | 0.714     | 0.714  | 0.714    | 7.0     |
| Excellence          | 1.000     | 0.833  | 0.909    | 6.0     |
| Private Traits      | 1.000     | 1.000  | 1.000    | 8.0     |
| Professional Traits | 0.900     | 0.818  | 0.857    | 11.0    |

## 3. 冲突矩阵 (Confusion Matrix)

行: Label 1, 列: Label 2

| label_1             | Agentic | Communal | Effort | Excellence | Private Traits | Professional Traits |
|:------------------- | -------:| --------:| ------:| ----------:| --------------:| -------------------:|
| Agentic             | 9       | 0        | 0      | 0          | 0              | 0                   |
| Communal            | 0       | 4        | 0      | 0          | 0              | 0                   |
| Effort              | 1       | 0        | 5      | 0          | 0              | 1                   |
| Excellence          | 1       | 0        | 0      | 5          | 0              | 0                   |
| Private Traits      | 0       | 0        | 0      | 0          | 8              | 0                   |
| Professional Traits | 0       | 0        | 2      | 0          | 0              | 9                   |

## 4. 差异仲裁与最终定稿 (Adjudication)

以下是所有标注不一致的词汇及其最终仲裁结果：

| word | label_1             | label_2             | final_label         | reason                          |
|:---- |:------------------- |:------------------- |:------------------- |:------------------------------- |
| 成就   | Excellence          | Agentic             | Excellence          | 强调结果性成就 (Excellence > Agentic)  |
| 技能   | Professional Traits | Effort              | Professional Traits | 强调职业资产 (Professional > Effort)  |
| 经验   | Professional Traits | Effort              | Professional Traits | 强调职业积累 (Professional > Effort)  |
| 坚韧   | Effort              | Agentic             | Effort              | 强调过程执行而非单纯掌控 (Effort > Agentic) |
| 效率   | Effort              | Professional Traits | Effort              | 强调过程投入与执行 (Effort)              |

## 5. 结论

最终生成的黄金标准数据集包含 45 个词汇，已保存至 `labeled_results_gold_final.csv`。