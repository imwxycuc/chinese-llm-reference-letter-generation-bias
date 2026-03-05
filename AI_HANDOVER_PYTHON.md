# Python 项目交接指南 (AI_HANDOVER_PYTHON.md)

这份文档旨在帮助你在新电脑上快速恢复并继续本项目的工作。

## 1. 环境管理

本项目使用标准的 Python `venv` 虚拟环境。

*   **推荐 Python 版本**: `>= 3.10` (当前环境为 Python 3.10)
*   **环境初始化命令**:

    ```bash
    # 1. 创建虚拟环境
    python -m venv .venv

    # 2. 激活虚拟环境
    # Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1
    # Windows (CMD):
    .\.venv\Scripts\activate.bat
    # Linux/Mac:
    source .venv/bin/activate

    # 3. 升级 pip
    python -m pip install --upgrade pip
    ```

## 2. 依赖清单

核心依赖库已整理至 `requirements.txt`。

*   **核心库**:
    *   `pandas`: 数据处理与 CSV 读写
    *   `transformers`: 调用 Hugging Face 模型 (如 BERT, XLM-Roberta)
    *   `torch`: 深度学习框架 (CPU/GPU 推理)
    *   `tqdm`: 进度条显示
    *   `zhipuai`: 智谱 AI SDK (用于生成任务)
    *   `openai`: OpenAI SDK (部分生成任务兼容)

*   **安装依赖**:

    ```bash
    pip install -r requirements.txt
    ```

## 3. 核心入口与项目结构

*   **项目根目录**: `C:\工作\论文\大模型性别偏见\数据分析与知识发现\code1-30` (在新电脑上请调整为实际路径)
*   **PYTHONPATH**: 通常不需要额外设置，确保在项目根目录下运行脚本即可。

### 主要脚本说明
*   **`classifier.py`**:
    *   **功能**: 对生成的信件进行“正式度 (formality)”和“情感 (sentiment)”分类评估。
    *   **关键参数**: `--input_file`, `--task both`, `-m zhipuai` (模型类型)。
    *   **并行支持**: 支持 `--limit` 和 `--offset` 参数进行手动分片。
*   **`hallucination_detection.py`**:
    *   **功能**: 检测生成内容中的幻觉 (Hallucination)，使用 `xlm-roberta-large-xnli` 模型。
    *   **关键参数**: `--num_shards`, `--shard_id`, `--model_type zhipuai`。
*   **`merge_classifier_parts.py` / `merge_zhipuai_hallucination_parts.py`**:
    *   **功能**: 用于合并并行运行产生的 `_partX.csv` 分片文件。

## 4. 当前开发状态 (截至 2026-02-02)

*   **最近修改**:
    *   修复了 `classifier.py` 在断点续传时的逻辑问题。
    *   创建了 `merge_classifier_parts.py` 和 `merge_zhipuai_hallucination_parts.py` 用于处理并行任务结果。
    *   清理了 `generated_letters\zhipuai\cbg\cbg_zhipuai_letters.csv` 中的重复数据 (0 duplicates confirmed)。
*   **运行状态**:
    *   **已完成**: `classifier.py` 全量运行并合并完毕 (5572 行)。
    *   **进行中**: `hallucination_detection.py` 正在以 4 个分片并行运行 (Shard 0-3)，预计需要数小时 (CPU 推理较慢)。
*   **下一步待办**:
    1.  等待 `hallucination_detection.py` 的 4 个分片任务完成。
    2.  运行 `python merge_zhipuai_hallucination_parts.py` 合并结果。
    3.  基于合并后的 `cbg_zhipuai_letters-eval_hallucination.csv` 进行后续的数据分析 (t-test 等)。

## 5. 避雷指南 & 注意事项

1.  **硬件加速 (GPU)**:
    *   当前项目在无 CUDA 环境下运行 (`torch.cuda.is_available() == False`)，推理速度较慢。
    *   **建议**: 如果新电脑有 NVIDIA 显卡，请务必安装对应版本的 PyTorch CUDA 版本 (如 `pip install torch --index-url https://download.pytorch.org/whl/cu118`) 以获得 10x+ 的加速。

2.  **网络与模型下载**:
    *   脚本中已内置国内镜像加速配置：
        ```python
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        ```
    *   首次运行 `hallucination_detection.py` 或 `classifier.py` 时会自动下载模型 (如 `joeddav/xlm-roberta-large-xnli`)，请保持网络通畅。

3.  **并行策略 (Sharding)**:
    *   由于单核 CPU 处理太慢，目前采用了**手动分片**策略。
    *   如果需要重新运行大批量任务，建议继续使用 `start-process` (PowerShell) 或多终端并行运行，并配合 `--shard_id` 参数。

4.  **文件路径**:
    *   注意 Windows (`\`) 与 Linux (`/`) 的路径分隔符差异。代码中大部分使用了 `os.path.join`，但部分硬编码路径可能需要检查。

5.  **API Key**:
    *   如果涉及到调用在线大模型 (如 `generate_cbg.py` 调用智谱 API)，请检查是否需要配置环境变量或 `.env` 文件中的 API Key。
