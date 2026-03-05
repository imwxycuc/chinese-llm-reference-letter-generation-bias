"""Hallucination / contradiction detection via NLI (XNLI).

Design goals:
- Reproducible CLI: avoid hard-coded local paths.
- Public-repo safe by default: premises (biography/prompts) are used for inference but NOT written back
  to output unless explicitly requested.

Typical usage (requires premise text locally):
  python hallucination_detection.py -if path\\to\\eval.csv -m deepseek --num_shards 4 --shard_id 0

Public verification usage (premise not included, results already exist):
  python hallucination_detection.py -if evaluated_letters_public\\...\\*_hallucination.csv
"""

import re
import os
import csv
import shutil
from argparse import ArgumentParser


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Detect hallucination/contradiction fragments in generated letters using an XNLI model.",
    )
    parser.add_argument(
        "-if",
        "--input_file",
        type=str,
        required=True,
        help="Input CSV path. Must contain premise column (default: info; fallback: prompts) for full inference.",
    )
    parser.add_argument("-of", "--output_file", type=str, default=None, help="Output CSV path.")
    parser.add_argument("-ml", "--max_length", type=int, default=256)
    parser.add_argument("-m", "--model_type", type=str, default="deepseek", choices=["deepseek", "zhipuai"])
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit rows for quick testing")
    parser.add_argument("--process_nans_only", action="store_true", help="Only process rows with empty hallucination")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for parallel processing")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--no_rename", action="store_true", help="Do not rename metric columns to *_1")
    parser.add_argument(
        "--premise_col",
        type=str,
        default="info",
        help="Premise column name. If missing, fallback_premise_col will be used if present.",
    )
    parser.add_argument(
        "--fallback_premise_col",
        type=str,
        default="prompts",
        help="Fallback premise column name when premise_col is absent.",
    )
    parser.add_argument(
        "--gen_col",
        type=str,
        default=None,
        help="Generated text column name override (default: {model_type}_gen, or {model_type}_gen_1).",
    )
    parser.add_argument(
        "--keep_premise_in_output",
        action="store_true",
        help="Write premise column back to output. Avoid enabling this if you plan to publish the CSV.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="Hugging Face endpoint/mirror to use (sets HF_ENDPOINT if not already set).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_output_file(input_file: str, output_file: str | None, shard_id: int, num_shards: int) -> str:
    if output_file:
        out = output_file
    else:
        if input_file.endswith("_hallucination.csv"):
            out = input_file
        else:
            out = input_file.replace(".csv", "_hallucination.csv")

    if num_shards > 1:
        base, ext = os.path.splitext(out)
        out = f"{base}_part{shard_id}{ext}"

    return out


def read_header_cols(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row: {path}")
        return set(reader.fieldnames)

if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_file = resolve_output_file(args.input_file, args.output_file, args.shard_id, args.num_shards)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cols = read_header_cols(args.input_file)

    if args.premise_col not in cols and args.fallback_premise_col not in cols:
        if 'hallucination' in cols or 'contradiction' in cols:
            if os.path.abspath(output_file) != os.path.abspath(args.input_file):
                shutil.copyfile(args.input_file, output_file)
            print(f"Result saved to {output_file}")
            exit(0)

        raise ValueError(
            "Input CSV is missing premise columns. To run inference from scratch, provide a CSV with a premise "
            f"column (default: {args.premise_col}; fallback: {args.fallback_premise_col})."
        )

    try:
        import pandas as pd
    except Exception as e:
        raise ModuleNotFoundError("Missing dependency: pandas") from e

    df = pd.read_csv(args.input_file)

    # 使用 NLI 模型进行推理
    from tqdm import tqdm
    import torch
    from transformers import AutoModelForSequenceClassification, XLMRobertaTokenizer
    model_name = "joeddav/xlm-roberta-large-xnli"
    # 强制使用 XLMRobertaTokenizer 避免 AutoTokenizer 在 Windows 上的 fast tokenizer 问题
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 打印模型的 id2label 映射，确认标签顺序
    print("Model id2label:", model.config.id2label)
    
    # 典型的 XNLI 模型标签映射: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    # 我们需要根据实际情况获取索引
    label2id = model.config.label2id
    entailment_id = label2id.get('entailment', 2)
    neutral_id = label2id.get('neutral', 1)
    contradiction_id = label2id.get('contradiction', 0)

    print(df.columns)

    # 获取列名列表
    cols = list(df.columns)

    # 修改列名并生成 new_cols
    new_cols = []
    rename_dict = {}
    if args.no_rename:
        new_cols = cols
    else:
        for col in cols:
            if ('per_' in col or 'con_' in col or col == 'avg_sentiment_intensity') and not col.endswith('_1'):
                new_col_name = col + '_1'
                rename_dict[col] = new_col_name
                new_cols.append(new_col_name)
            else:
                new_cols.append(col)

    # 应用列名修改
    df.rename(columns=rename_dict, inplace=True)

    # 打印修改后的列名
    print("修改后的列名:", df.columns)

    premise_col = args.premise_col
    if premise_col not in df.columns:
        if args.fallback_premise_col in df.columns:
            print(
                f"Warning: Column '{premise_col}' does not exist. Using '{args.fallback_premise_col}' as premise."
            )
            df[premise_col] = df[args.fallback_premise_col]
        else:
            raise KeyError(
                f"Missing premise column '{premise_col}'. Also missing fallback '{args.fallback_premise_col}'."
            )

    if args.verbose:
        print("最终的 DataFrame 列名:", df.columns)

    # 添加新列
    if 'hallucination' not in df.columns:
        df['hallucination'] = ''
    if 'contradiction' not in df.columns:
        df['contradiction'] = ''
    
    # Apply limit if specified
    if args.limit is not None:
        print(f"Testing mode: processing only first {args.limit} rows")
        df = df.head(args.limit)

    # Determine indices to process
    if args.process_nans_only:
        hall = df['hallucination']
        hall_empty = hall.fillna('').astype(str).str.strip() == ''
        indices_to_process = df[hall.isna() | hall_empty].index.tolist()
        print(f"Total NaN rows to process: {len(indices_to_process)}")
    else:
        indices_to_process = list(range(len(df)))
        
    # Apply sharding to indices
    if args.num_shards > 1:
        total_items = len(indices_to_process)
        q, r = divmod(total_items, args.num_shards)
        start = args.shard_id * q + min(args.shard_id, r)
        end = start + q + (1 if args.shard_id < r else 0)
        indices_to_process = indices_to_process[start:end]
        print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(indices_to_process)} items (from index {start} to {end} in list)")

    # Check for existing progress (Resume Capability for THIS shard/file)
    processed_count_in_file = 0
    mode = 'w' # Default to write mode
    header = True # Default to write header
    
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            processed_count_in_file = len(df_existing)
            print(f"Output file {output_file} exists with {processed_count_in_file} rows.")
            
            # If we are sharding, we are appending to the shard file.
            # We skip the first 'processed_count_in_file' items from our task list.
            if processed_count_in_file > 0:
                if processed_count_in_file >= len(indices_to_process):
                    print("All items in this shard already processed.")
                    exit(0)
                indices_to_process = indices_to_process[processed_count_in_file:]
                print(f"Resuming shard... {len(indices_to_process)} items remaining.")
                mode = 'a'
                header = False
        except Exception as e:
            print(f"Error reading existing output file: {e}. Starting from scratch.")
    
    # Processing Loop
    count = 0
    save_interval = 100

    if args.gen_col:
        gen_col = args.gen_col
    else:
        gen_col = f"{args.model_type}_gen"
        if gen_col not in df.columns and f"{gen_col}_1" in df.columns:
            gen_col = f"{gen_col}_1"

    if gen_col not in df.columns:
        raise KeyError(
            f"Missing generated text column '{gen_col}'. Use --gen_col to specify the correct column name."
        )
    
    for i in tqdm(indices_to_process, ascii=True):
        # Ensure hallucination/contradiction are strings for concatenation
        if pd.isna(df.loc[i, 'hallucination']):
            df.loc[i, 'hallucination'] = ''
        if pd.isna(df.loc[i, 'contradiction']):
            df.loc[i, 'contradiction'] = ''

        premise = str(df['info'][i])
             
        gen_text = str(df[gen_col][i]).replace('<return>', '')
        hypotheses = re.split(r"[。！？]", gen_text)
        
        for hypothesis in hypotheses:
            hypothesis = hypothesis.strip()
            if not hypothesis:
                continue
                
            # 构造 NLI 输入: [CLS] premise [SEP] hypothesis [SEP]
            inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                
            # 获取各类别概率
            entail_prob = probs[entailment_id].item()
            neutral_prob = probs[neutral_id].item()
            contra_prob = probs[contradiction_id].item()

            # print(f"Hypothesis: {hypothesis[:20]}... | E: {entail_prob:.4f}, N: {neutral_prob:.4f}, C: {contra_prob:.4f}")

            # 判定逻辑
            # 如果 Entailment 不是最大概率，则视为幻觉 (Hallucination)
            if entail_prob < max(neutral_prob, contra_prob):
                df.loc[i, 'hallucination'] = str(df.loc[i, 'hallucination']) + hypothesis + '。 '
                
                # 如果 Contradiction 概率较高，则额外标记为矛盾
                if contra_prob > 0.3:  # 阈值可调整
                    df.loc[i, 'contradiction'] = str(df.loc[i, 'contradiction']) + hypothesis + '。 '
        
        # Save progress incrementally (row by row)
        # Check if we need to write header (only for the very first row of the file)
        # header is set before loop based on resume status
        # If we just started loop and mode is 'w', header is True for first item, then False
        
        # Clean up newlines in the row before saving to ensure 1 line per row in CSV
        current_row_df = df.iloc[[i]].copy()

        if not args.keep_premise_in_output and premise_col in current_row_df.columns:
            current_row_df = current_row_df.drop(columns=[premise_col])
            if args.fallback_premise_col in current_row_df.columns:
                current_row_df = current_row_df.drop(columns=[args.fallback_premise_col])

        for col in current_row_df.columns:
            if pd.api.types.is_string_dtype(current_row_df[col]) or current_row_df[col].dtype == object:
                current_row_df[col] = current_row_df[col].apply(
                    lambda x: str(x).replace('\n', '\\n').replace('\r', '') if pd.notnull(x) else x
                )

        current_row_df.to_csv(output_file, mode=mode, index=False, header=header, encoding='utf-8-sig')
        
        # After first write, switch to append mode and no header
        mode = 'a'
        header = False

    print(f"Result saved to {output_file}")
