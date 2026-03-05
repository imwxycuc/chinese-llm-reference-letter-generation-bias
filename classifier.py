import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"  # 增加超时时间到30秒

import pandas as pd
from transformers import pipeline
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
from tqdm.auto import tqdm  # 使用tqdm.auto以自动适应不同的环境
import re
tqdm.pandas()  # 初始化tqdm_pandas，将progress_apply添加到DataFrame和Series对象上

import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# PATH_TO_COLS = {
#     "df_f_acting_2_para_zhipuai.csv": ("gender", "zhipuai_gen"),
# }
PATH_TO_COLS = {
    "df_f_acting_2_para_deepseek.csv": ("gender", "deepseek_gen"),
}
task_label_mapping = {
    # "sentiment": ("LABEL_1", "LABEL_0"), # 全分类任务不再使用此映射
    "formality": ("LABEL_1", "LABEL_0"),
}

def predict(text, classifier, task, is_sentencelevel=True, return_distribution=False):
    if not text or not isinstance(text, str):
        print(f"Warning: Invalid input text: {text}")
        if return_distribution:
            return []
        return ("UNKNOWN", 0.0)
    
    result = None
    labels = []
    scores = []
    distributions = []
    max_length = 512

    if is_sentencelevel:
        # 使用更完善的标点符号分句
        sentences = re.split(r'[。！？；]', text)
        sentences = [s for s in sentences if s.strip()] # 去除空字符串
        
        for sentence in sentences:
            if not sentence: continue
            
            # 处理超长句子
            if len(sentence) >= max_length:
                # 简单截断或分块处理，这里保持原有分块逻辑
                for i in range(0, len(sentence), max_length):
                    chunk = sentence[i:i+max_length]
                    if return_distribution:
                        # top_k=None 返回所有类别的概率分布
                        res = classifier(chunk + "。", truncation=True, top_k=None)
                        distributions.append(res) # res 是一个列表的列表（因为输入是单个字符串），或者直接是列表？
                        # pipeline 对单个字符串输入，返回 list of dicts。
                        # 对 list 输入，返回 list of list of dicts。
                        # 这里输入是 string，所以返回 list of dicts。
                    else:
                        res = classifier(chunk + "。", truncation=True)[0]
                        labels.append(res["label"])
                        scores.append(res["score"])
            else:
                if return_distribution:
                    res = classifier(sentence + "。", truncation=True, top_k=None)
                    distributions.append(res)
                else:
                    res = classifier(sentence + "。", truncation=True)[0]
                    labels.append(res["label"])
                    scores.append(res["score"])
    else:
        # 文档级 (非分句)
        if len(text) >= max_length:
            for i in range(0, len(text), max_length):
                chunk = text[i:i+max_length]
                if return_distribution:
                    res = classifier(chunk, truncation=True, top_k=None)
                    distributions.append(res)
                else:
                    res = classifier(chunk, truncation=True)[0]
                    labels.append(res["label"])
                    scores.append(res["score"])
        else:
            if return_distribution:
                res = classifier(text, truncation=True, top_k=None)
                distributions.append(res)
            else:
                res = classifier(text, truncation=True)[0]
                labels.append(res["label"])
                scores.append(res["score"])
    
    if return_distribution:
        return distributions
    return labels, scores

def calculate_percentages_and_confidences(labels, scores, positive_label):
    label_count = Counter(labels)
    positive_count = label_count[positive_label]
    total_count = sum(label_count.values())
    percentage = positive_count / total_count if total_count else 0
    confidence = sum(scores) / len(scores) if scores else 0
    return percentage, confidence

def calculate_full_sentiment_stats_distribution(distributions):
    """
    基于全概率分布计算情感统计量
    distributions: list of list of dicts. 每个元素是一个句子的预测分布。
    Example: [[{'label': '1 star', 'score': 0.1}, {'label': '5 stars', 'score': 0.9}], ...]
    """
    if not distributions:
        return {f"stars{i}_freq": 0 for i in range(1, 6)} | {"avg_sentiment_intensity": 0.0}

    doc_expectations = [] # 存储每个句子的期望值 E(s_i)
    all_top_scores = [] # 存储每个句子的最高分对应的分值 (用于频次统计)

    # Debug: 打印第一个分布以确认格式
    if distributions and not hasattr(calculate_full_sentiment_stats_distribution, "debug_printed"):
        print(f"DEBUG: First distribution encountered: {distributions[0]}")
        calculate_full_sentiment_stats_distribution.debug_printed = True

    for dist in distributions:
        # dist 是一个句子的预测结果列表，如 [{'label': 'LABEL_0', 'score': 0.9}, ...]
        
        # 1. 计算单句期望值 E(s_i) = sum(k * P(k))
        sentence_expectation = 0.0
        
        # 寻找该句子的Top-1标签用于频次统计
        best_label = None
        best_prob = -1.0
        
        for item in dist:
            label = item['label']
            prob = item['score']
            
            # 解析分值 k
            try:
                k = int(re.search(r'\d+', label).group())
                # 映射逻辑
                if "LABEL" in label.upper():
                    # 假设 LABEL_0 -> 1分, LABEL_4 -> 5分
                    k = k + 1
                elif "star" in label.lower():
                    # 假设 "1 star" -> 1分
                    pass 
                
                # 确保 k 在 1-5 之间 (容错)
                k = max(1, min(5, k))
                
                # 累加期望
                sentence_expectation += k * prob
                
                # 更新Top-1
                if prob > best_prob:
                    best_prob = prob
                    best_label = k
                    
            except (AttributeError, ValueError):
                continue
        
        doc_expectations.append(sentence_expectation)
        if best_label is not None:
            all_top_scores.append(best_label)

    # 2. 频次统计 (基于 Top-1)
    counts = Counter(all_top_scores)
    stats = {f"stars{i}_freq": counts.get(i, 0) for i in range(1, 6)}
    
    # 3. 文档级得分聚合 (算术平均)
    if doc_expectations:
        stats["avg_sentiment_intensity"] = sum(doc_expectations) / len(doc_expectations)
    else:
        stats["avg_sentiment_intensity"] = 0.0
        
    return stats

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-if", "--input_file", type=str, default=None)
    parser.add_argument("-of", "--output_folder", type=str, default="./result/Language Style")
    parser.add_argument("-t", "--task", type=str, default="both", choices=["sentiment", "formality", "both"])
    parser.add_argument("-m", "--model_type", type=str, default="deepseek")
    parser.add_argument("--output_type", type=str, default="csv")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of rows to process for testing")
    parser.add_argument("-off", "--offset", type=int, default=0, help="Offset to start processing from")
    parser.add_argument("-os", "--output_suffix", type=str, default="", help="Suffix for output filename")
    parser.add_argument("-tc", "--text_column", type=str, default=None, help="The column name containing the text to analyze")
    args = parser.parse_args()

    # Determine input file and text column based on model_type
    input_file = args.input_file
    text_column = args.text_column
    
    if input_file is None:
        if args.model_type == 'zhipuai':
            input_file = "./generated_letters/zhipuai/clg/clg_letters.csv"
            if text_column is None:
                text_column = "zhipuai_gen"
        elif args.model_type == 'deepseek':
            input_file = "./generated_letters/deepseek/clg/clg_deepseek_letters.csv"
            if text_column is None:
                text_column = "deepseek_gen"
        else:
            raise ValueError("Invalid model_type. Choose 'zhipuai' or 'deepseek'.")
    else:
        # If input file is provided but text_column is not, try to infer or error
        if text_column is None:
            if args.model_type == 'zhipuai':
                text_column = "zhipuai_gen"
            elif args.model_type == 'deepseek':
                text_column = "deepseek_gen"
            else:
                 # Default fallback if model_type doesn't match
                 text_column = "text" 
    
    print(f"Processing file: {input_file}")
    print(f"Target text column: {text_column}")

    data_path = input_file
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    INPUT = text_column

    # 读取 CSV 文件
    df = pd.read_csv(input_file, encoding='utf-8')

    # 应用 offset 和 limit 参数进行分片
    if args.offset > 0:
        print(f"Applying offset: {args.offset}")
        df = df.iloc[args.offset:]
    
    if args.limit is not None:
        print(f"Processing limit: {args.limit} rows.")
        df = df.head(args.limit)

    classifier_sentiment = None
    classifier_formality = None
    if args.task == "sentiment" or args.task == "both":
        # classifier_sentiment = pipeline("sentiment-analysis", model="bert-base-chinese")
        # 切换到全分类模型
        classifier_sentiment = pipeline("text-classification", model="uer/roberta-base-finetuned-jd-full-chinese")
    if args.task == "formality" or args.task == "both":
        # classifier_formality = pipeline("text-classification", "bert-base-chinese")
        classifier_formality = pipeline("text-classification", "./models/formality_classifier/best_model")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    output_filename = file_name + args.output_suffix + '-eval.csv'
    output_path = os.path.join(args.output_folder, output_filename)
    
    start_index = 0
    overwrite_mode = False
    
    if args.output_type == "csv" and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if len(existing_df) == len(df):
                print(f"Found complete existing output file. Loading it to append missing columns (e.g., formality).")
                df = existing_df
                start_index = 0
                overwrite_mode = True
            else:
                start_index = len(existing_df)
                print(f"Found existing output file (partial). Resuming from index {start_index}.")
        except Exception as e:
            print(f"Error reading existing output file: {e}. Starting from scratch.")
    
    if start_index >= len(df) and not overwrite_mode:
        print("All data already processed.")
    else:
        chunk_size = 32
        print(f"Processing rows {start_index} to {len(df)}...")
        
        # 使用tqdm显示总进度，initial参数设置初始进度
        with tqdm(total=len(df), initial=start_index, unit="rows", desc="Total Progress") as pbar:
            for i in range(start_index, len(df), chunk_size):
                try:
                    chunk = df.iloc[i : i + chunk_size].copy()
                    
                    if args.task == "formality" or args.task == "both":
                        # 检查是否已存在结果，避免重复计算
                        if "per_for" not in chunk.columns or chunk["per_for"].isnull().any() or overwrite_mode:
                            # 如果是overwrite_mode且列已存在，我们可能想跳过，但为了安全（也许想更新），
                            # 这里我们优化一下：如果列存在且不为空，就跳过
                            if "per_for" in chunk.columns and not chunk["per_for"].isnull().any():
                                pass # Skip computation
                            else:
                                formality_results = chunk[INPUT].progress_apply(
                                    lambda x: predict(x, classifier_formality, "formality")
                                )
                                per_for, con_for = zip(*formality_results.apply(
                                    lambda x: calculate_percentages_and_confidences(x[0], x[1], task_label_mapping["formality"][0])
                                ))
                                chunk["per_for"] = per_for
                                chunk["con_for"] = con_for

                    if args.task == "sentiment" or args.task == "both":
                        # 检查是否已存在结果
                        if "avg_sentiment_intensity" not in chunk.columns or chunk["avg_sentiment_intensity"].isnull().any():
                             # 同样，如果列存在且完整，跳过
                             pass
                             # 但由于这里代码结构需要重构才能优雅跳过（因为stats_df的合并），
                             # 我们简单检查一下关键列
                        
                        if "avg_sentiment_intensity" not in chunk.columns or chunk["avg_sentiment_intensity"].isnull().any():
                            # 获取全概率分布
                            sentiment_results = chunk[INPUT].apply(
                                lambda x: predict(x, classifier_sentiment, "sentiment", return_distribution=True)
                            )
                            # 使用基于分布的统计函数计算
                            stats_list = sentiment_results.apply(
                                lambda x: calculate_full_sentiment_stats_distribution(x)
                            )
                            
                            # 将字典列表转换为DataFrame列
                            stats_df = pd.DataFrame(stats_list.tolist(), index=chunk.index)
                            
                            # 将新列合并到chunk中
                            for col in stats_df.columns:
                                chunk[col] = stats_df[col]

                    if args.output_type == "csv":
                        mode = 'a'
                        header = False
                        
                        if overwrite_mode:
                            # 如果是全量更新模式
                            if i == 0:
                                mode = 'w'
                                header = True
                            else:
                                mode = 'a'
                                header = False
                        else:
                            # 断点续传模式
                            if i == 0: # 只有在从头开始时才写header
                                mode = 'w'
                                header = True
                        
                        chunk.to_csv(output_path, mode=mode, header=header, index=False)
                        pbar.update(len(chunk))
                except Exception as e:
                    error_msg = f"Error processing chunk starting at index {i}: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
                    # 继续处理下一个chunk，不中断程序
                    pbar.update(len(chunk)) # 即使出错也要更新进度
                    continue
                
        print("Finished output of percent/confidence to {}".format(output_path))
