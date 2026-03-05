# 数据来源与使用说明（Data Sources & Usage）

本仓库用于复现与复核论文实验结果，遵循“可复现、可复核，但不直接再分发第三方原始数据”的公开原则：能在本仓库内完成统计复核与结果重算；如需从头生成数据（尤其是人物简介/前提信息），请按本文档指引自行从原始来源获取。

## 1. 第三方原始信息来源

### 1.1 人物简介/前提信息（Biography / Premise）

- **来源仓库**：`uclanlp/biases-llm-reference-letters`  
  - URL： https://github.com/uclanlp/biases-llm-reference-letters
- **许可协议**：MIT License（原仓库声明：Copyright (c) 2024 Natural Language Processing @UCLA）
- **本仓库的处理方式**：
  - 不在本仓库中直接包含/发布该来源的原始人物简介文本、姓名等字段。
  - 与人物简介直接相关的字段（如 `info`、`career_sec`、`personal_sec`、`first_name`、`last_name`、`seed_*`）在公开数据中已被移除或不提供。
- **复现者如需从头生成推荐信或重跑幻觉检测**：
  - 请自行从上述原仓库下载人物简介数据，并在本地按脚本约定放置到相应目录（例如 `biography_dataset/` 或脚本参数指定的位置）。
  - 本仓库公开版本中不包含 `biography_dataset/` 目录（或等价的原始人物简介文件），属于刻意省略以避免再分发第三方原始文本。

## 2. 本仓库公开数据的组成与含义

### 2.1 生成推荐信（Model-generated Letters）

目录：`generated_letters/`

- `generated_letters/deepseek/cbg/cbg_deepseek_letters.csv`  
- `generated_letters/zhipuai/cbg/cbg_zhipuai_letters.csv`  
- `generated_letters/deepseek/clg/clg_deepseek_letters.csv`  
- `generated_letters/zhipuai/clg/clg_zhipuai_letters.csv`

字段说明（以 CBG 为例，CLG 类似）：

- `row_id`：行号（本仓库添加，用于稳定对齐与抽样，不含任何原始身份信息）
- `gender`：性别标签（例如 `m/f` 或 `male/female`，以文件实际取值为准）
- `occupation`：职业类别标签
- `{model}_gen`：模型生成的推荐信正文，例如 `deepseek_gen`、`zhipuai_gen`

说明：

- 这些推荐信文本为模型生成内容，不包含第三方人物简介原文。
- 生成脚本依赖外部 API（见“3. 使用方式”），本仓库不提供任何 API 密钥文件。

### 2.2 评估与统计输入（Public Evaluation Tables）

目录：`evaluated_letters_public/`

该目录下 CSV 用于复核论文中的统计、显著性检验与可视化，属于“公开可复核”的最小数据集。其核心特点是：

- 保留：性别/职业标签、风格与情感等统计指标、幻觉检测输出（如已计算）
- 不保留：第三方人物简介文本、姓名字段、生成时的 seed 等可能导致回溯原始数据的字段

常见文件（示例）：

- `evaluated_letters_public/*/clg_*_letters-eval.csv`
- `evaluated_letters_public/*/cbg_*_letters-eval.csv`
- `evaluated_letters_public/*/cbg_*_letters-eval_hallucination.csv`
- `evaluated_letters_public/*/cbg_*_letters-eval_hallucination-eval.csv`

字段说明（不同文件略有差异，以表头为准）：

- `per_for` / `con_for`：正式性指标（来自 `classifier.py` 的 formality 模型）
- `avg_sentiment_intensity`、`stars1_freq`~`stars5_freq`：情感强度与星级统计（来自 `classifier.py` 的情感模型）
- `hallucination` / `contradiction`：基于 NLI 的幻觉/矛盾片段抽取结果（来自 `hallucination_detection.py`）
- `*_gen`：生成列（为便于复核与二次分析，CBG 的幻觉检测结果文件中包含 `deepseek_gen` / `zhipuai_gen`）
- 带 `_1` 后缀的指标列：用于“原始 vs 幻觉版本”对比的同名指标列

## 3. 使用方式（从本仓库复核/从头复现）

### 3.1 复核统计结果（不需要第三方原始人物简介）

目标：在不下载第三方人物简介数据的情况下，复核论文中的统计与显著性检验。

- 词汇匹配统计：运行 [biases_string_matching.py](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/biases_string_matching.py)（输入可用 `generated_letters/**` 或对应的已计算结果）
- 风格/情感指标：使用 [classifier.py](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/classifier.py) 生成或直接读取 `evaluated_letters_public/` 内的 `*-eval.csv`
- 显著性检验：使用 [ttest.py](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/ttest.py) 对指标列进行检验

依赖见 [requirements.txt](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/requirements.txt)。

### 3.2 从头生成推荐信（需要第三方人物简介 + API 权限）

目标：从人物简介出发重新生成 CBG/CLG 推荐信文本。

- 人物简介：请自行从 `uclanlp/biases-llm-reference-letters` 获取并放置到脚本所需目录
- 生成脚本：见 [generate_cbg.py](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/generate_cbg.py)、[generate_clg.py](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/generate_clg.py)
- API 密钥：本仓库不包含任何密钥文件。请在本地自行配置，并避免将密钥提交到公开仓库。
- 重要提示：如果你打算公开仓库，请不要把人物简介原文或包含 `info/career_sec/personal_sec` 等列的中间 CSV 一并上传。

### 3.3 幻觉检测（Hallucination Detection）

脚本： [hallucination_detection.py](file:///c:/Users/12905/OneDrive/%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%9F%A5%E8%AF%86%E5%8F%91%E7%8E%B0/code1-30/hallucination_detection.py)

- 算法：基于 NLI（XNLI）模型对“前提（人物简介）-假设（生成句子）”进行蕴含关系判定，抽取可能的幻觉/矛盾片段。
- 模型：`joeddav/xlm-roberta-large-xnli`（脚本中定义）
- 公开复核模式：当输入文件缺少 `info/prompts` 但已包含 `hallucination/contradiction` 时，脚本可直接保存结果文件用于复核，不强制下载模型或依赖完整推理环境。

## 4. 结果文件与复核路径（建议引用）

目录：`result/`

- `result/lexical_analysis/`：词汇偏见统计与报告
- `result/Language Style/`：风格/情感指标与显著性检验结果
- `result/hallucination_detection/`：幻觉检测指标与显著性检验结果

如需在论文或补充材料中引用本仓库数据，请同时引用：

- 本仓库（含版本/提交号）
- 第三方人物简介来源仓库 `uclanlp/biases-llm-reference-letters`（MIT License）

## 5. 合规与隐私说明

- 本仓库不发布第三方人物简介原文、姓名字段与生成所用 seed 信息。
- 本仓库不包含任何 API 密钥与个人凭据；如需复现实验，请在本地自行配置。
