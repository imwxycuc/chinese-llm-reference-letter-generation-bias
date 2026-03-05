import sys
import pandas as pd
from generation_util import *
import random
import os
from tqdm import tqdm
import argparse
from zhipuai.core._errors import APIRequestFailedError
from openai import OpenAI
import json
import logging

# def zhipuai_gen(occu, gend, output_folder):
#     if gend == "f":
#         csv_file = f"./biography_dataset/preprocessed_bios/df_f_{occu}_2_para.csv"
#     else:
#         csv_file = f"./biography_dataset/preprocessed_bios/df_m_{occu}_2_para.csv"
#     file_name = csv_file.split('/')[-1].split('.')[0] + '_zhipuai.csv'
#     output_file_path = os.path.join(output_folder, file_name)
#
#     if not os.path.exists(csv_file):
#         raise Exception(f"Occupation {occu} for ZhipuAi has not been generated yet!")
#    
#     if occu == "acting":
#         real_occupation = "actor"
#     else:
#         real_occupation = occu.rstrip("s")
#
#     df = pd.read_csv(csv_file)
#     if "info" not in list(df.columns) or "first_name" not in list(df.columns):
#         raise Exception("info and name must be in df's columns.")
#     df["zhipuai_gen"] = -1
#
#     # 如果输出文件已存在，读取已有数据
#     if os.path.exists(output_file_path):
#         existing_df = pd.read_csv(output_file_path)
#     else:
#         existing_df = pd.DataFrame(columns=df.columns)
#
#     for i, row in tqdm(df.iterrows(), ascii=True):
#         pronoun = "他" if row["gender"] == "m" else "她"
#         generated_response = generate_response_rec_zhipuai(
#             {
#                 "occupation": real_occupation,
#                 "name": "{} {}".format(row["first_name"], row["last_name"]),
#                 "pronoun": pronoun,
#                 "info": row["info"],
#             }
#         )
#         generated_response = generated_response.replace("\n", "<return>")
#         df.at[i, "zhipuai_gen"] = generated_response
#
#         # 更新或添加当前行的数据到 existing_df
#         if i in existing_df.index:
#             existing_df.at[i, "zhipuai_gen"] = generated_response
#         else:
#             existing_df = pd.concat([existing_df, df.iloc[[i]]], ignore_index=True)
#
#         # 每生成一份推荐信后立即保存到文件
#         existing_df.to_csv(output_file_path, index=False)
#
#     return

def deepseek_gen(occu, gend, output_folder):
    if gend == "f":
        csv_file = f"./biography_dataset/preprocessed_bios/df_f_{occu}_2_para.csv"
    else:
        csv_file = f"./biography_dataset/preprocessed_bios/df_m_{occu}_2_para.csv"
    file_name = csv_file.split('/')[-1].split('.')[0] + '_deepseek.csv'
    output_file_path = os.path.join(output_folder, file_name)

    if not os.path.exists(csv_file):
        raise Exception(f"Occupation {occu} for DeepSeek has not been generated yet!")
    
    if occu == "acting":
        real_occupation = "actor"
    else:
        real_occupation = occu.rstrip("s")

    df = pd.read_csv(csv_file)
    if "info" not in list(df.columns) or "first_name" not in list(df.columns):
        raise Exception("info and name must be in df's columns.")
    df["deepseek_gen"] = -1

    # 如果输出文件已存在，读取已有数据
    if os.path.exists(output_file_path):
        existing_df = pd.read_csv(output_file_path)
    else:
        existing_df = pd.DataFrame(columns=df.columns)

    for i, row in tqdm(df.iterrows(), ascii=True):
        # 检查是否已经存在生成的推荐信
        if i in existing_df.index and existing_df.at[i, "deepseek_gen"] != -1:
            print(f"Recommendation letter for {row['first_name']} {row['last_name']} already exists, skipping...")
            continue

        pronoun = "他" if row["gender"] == "m" else "她"
        utt = RECLETTER_PROMPTS[0].format(
            real_occupation,
            "{} {}".format(row["first_name"], row["last_name"]),
            pronoun,
            row["info"]
        )
        # 直接调用generate_deepseek函数
        generated_response = generate_deepseek(utt)
        generated_response = generated_response.replace("\n", "<return>")
        df.at[i, "deepseek_gen"] = generated_response

        # 更新或添加当前行的数据到 existing_df
        if i in existing_df.index:
            existing_df.at[i, "deepseek_gen"] = generated_response
        else:
            existing_df = pd.concat([existing_df, df.iloc[[i]]], ignore_index=True)

        # 每生成一份推荐信后立即保存到文件
        existing_df.to_csv(output_file_path, index=False)

    return

def model_gen(occu, gend, model_type, output_folder):
    if gend == "f":
        csv_file = f"./biography_dataset/preprocessed_bios/df_f_{occu}_2_para.csv"
    else:
        csv_file = f"./biography_dataset/preprocessed_bios/df_m_{occu}_2_para.csv"
    file_name = csv_file.split('/')[-1].split('.')[0] + '_{}.csv'.format(model_type)
    output_file_path = os.path.join(output_folder, file_name)

    if not os.path.exists(csv_file):
        raise Exception(f"Occupation {occu} for Model {model_type} has not been generated yet!")
    
    if occu == "acting":
        real_occupation = "actor"
    else:
        real_occupation = occu.rstrip("s")

    if model_type == "alpaca":
        tokenizer = LlamaTokenizer.from_pretrained(
            "chavinlo/alpaca-native", model_max_length=1024
        )
        model = LlamaForCausalLM.from_pretrained("chavinlo/alpaca-native")

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)

    elif model_type == "vicuna":
        tokenizer = LlamaTokenizer.from_pretrained("/local/elaine1wan/vicuna")
        model = LlamaForCausalLM.from_pretrained("/local/elaine1wan/vicuna")

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)

    elif model_type == "stablelm":
        tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
        model = AutoModelForCausalLM.from_pretrained(
            "StabilityAI/stablelm-tuned-alpha-7b"
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)
        # model.to(device)

    elif model_type == "falcoln":
        model = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.half().to(device)
    else:
        raise NotImplementedError

    df = pd.read_csv(csv_file)
    if "info" not in list(df.columns) or "first_name" not in list(df.columns):
        raise Exception("info and name must be in df's columns.")
    df["{}_gen".format(model_type)] = -1
    print("Total generations: {}".format(len(df)))

    # 如果输出文件已存在，读取已有数据
    if os.path.exists(output_file_path):
        existing_df = pd.read_csv(output_file_path)
    else:
        existing_df = pd.DataFrame(columns=df.columns)

    write_amount = 0
    for i, row in tqdm(df.iterrows(), ascii=True):
        pronoun = "him" if row["gender"] == "m" else "her"
        if model_type == "alpaca":
            generated_response = generate_response_rec_alpaca(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        elif model_type == "vicuna":
            generated_response = generate_response_rec_vicuna(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        elif model_type == "stablelm":
            generated_response = generate_response_rec_stablelm(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        elif model_type == "falcoln":
            generated_response = generate_response_rec_falcon(
                {
                    "occupation": real_occupation,
                    "name": "{} {}".format(row["first_name"], row["last_name"]),
                    "pronoun": pronoun,
                    "info": row["info"],
                },
                model,
                tokenizer,
                device,
            )
        generated_response = generated_response.replace("\n", "<return>")
        df.at[i, "{}_gen".format(model_type)] = generated_response

        # 更新或添加当前行的数据到 existing_df
        if i in existing_df.index:
            existing_df.at[i, "{}_gen".format(model_type)] = generated_response
        else:
            existing_df = pd.concat([existing_df, df.iloc[[i]]], ignore_index=True)

        # 每生成一份推荐信后立即保存到文件
        existing_df.to_csv(output_file_path, index=False)

        write_amount += 1
    print("Number of generated samples: {}".format(write_amount))
    return


def siliconflow_gen_file(config_path, input_file, output_folder):
    # Load config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    sf_config = config['models']['siliconflow']
    api_key = sf_config['api_key']
    base_url = sf_config['base_url']
    model_name = sf_config['model_name']

    # Initialize client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Setup logging
    log_file = os.path.join(output_folder, 'generation.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also print errors to console
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Starting generation for {input_file} using {model_name}")

    # Read input file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        print(f"Failed to read input file: {e}")
        return

    # Prepare output file path
    input_filename = os.path.basename(input_file)
    output_filename = input_filename.replace('.csv', '_siliconflow.csv')
    if output_filename == input_filename: 
         output_filename = input_filename.replace('.csv', '_out.csv')
    output_file_path = os.path.join(output_folder, output_filename)

    # Check for existing progress (Resume capability)
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} exists. Resuming...")
        try:
            existing_df = pd.read_csv(output_file_path)
            # Merge existing results into df based on index
            # Ensure 'siliconflow_gen' column exists in existing_df
            if 'siliconflow_gen' in existing_df.columns:
                 # Map existing results to df
                 # Assuming index alignment is preserved
                 df['siliconflow_gen'] = existing_df['siliconflow_gen']
            else:
                 df['siliconflow_gen'] = -1
        except Exception as e:
            logging.error(f"Error reading existing output file: {e}")
            df['siliconflow_gen'] = -1
    else:
        df['siliconflow_gen'] = -1

    # Check if 'siliconflow_gen' column exists (if it wasn't added by resume block)
    if 'siliconflow_gen' not in df.columns:
        df['siliconflow_gen'] = -1

    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    
    processed_count = 0
    
    # Use tqdm for progress bar
    for i, row in tqdm(df.iterrows(), total=total_rows, ascii=True, desc="Processing"):
        # Check if already processed
        current_gen = row.get('siliconflow_gen', -1)
        # Check if current_gen is valid (not -1 and not empty/NaN)
        if pd.notna(current_gen) and str(current_gen) != "-1" and str(current_gen).strip() != "":
            continue

        # Prepare prompt
        occupation = row['occupation']
        if occupation == "acting":
            real_occupation = "actor"
        else:
            real_occupation = occupation.rstrip("s")
            
        pronoun = "他" if row["gender"] == "m" else "她"
        name = "{} {}".format(row["first_name"], row["last_name"])
        info = row["info"]
        
        # Construct prompt
        prompt = RECLETTER_PROMPTS[0].format(
            real_occupation,
            name,
            pronoun,
            info
        )

        try:
            generated_text = generate_text(client, model_name, prompt)
            generated_text = generated_text.replace("\n", "<return>")
            
            # Update DataFrame
            df.at[i, 'siliconflow_gen'] = generated_text
            processed_count += 1
            
            # Save progress immediately
            df.to_csv(output_file_path, index=False)
            
        except Exception as e:
            error_msg = f"Error processing row {i} ({name}): {e}"
            logging.error(error_msg)
            # print(error_msg) # console handler already prints errors
            continue

    print(f"Processing complete. {processed_count} new letters generated.")
    print(f"Output saved to {output_file_path}")


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="zhipuai", help="Model type.")
    parser.add_argument("--model", default="deepseek", help="Model type: deepseek, siliconflow, etc.")
    parser.add_argument("--n", type=int, default=1, help="Number of samples for each occupation for each gender.")
    parser.add_argument('-of', '--output_folder', default='./generated_letters/deepseek/cbg')
    parser.add_argument('--input_file', help="Specific input CSV file to process (required for siliconflow)")
    parser.add_argument('--config', default='config.json', help="Path to config.json")
    
    args = parser.parse_args()
    print(args)

     # 检查--n参数
    if args.n <= 0:
        print("The --n argument must be a positive integer.")
        sys.exit(1)

    if args.model == "siliconflow":
        if not args.input_file:
            # Fallback to default behavior if input_file not provided? 
            # Or enforce input_file as per user request context.
            # User specifically asked to process a specific file.
            print("Error: --input_file is required for siliconflow model in this mode.")
            return
        siliconflow_gen_file(args.config, args.input_file, args.output_folder)
        return

    for occupation in ['acting', 'chefs', 'artists', 'dancers', 'comedians', 'models', 'musicians', 'podcasters', 'writers', 'sports']:
        for gend in ["m", "f"]:
            try:
                # if args.model == "zhipuai":
                if args.model == "deepseek":
                    deepseek_gen(occupation, gend, args.output_folder)
                    # zhipuai_gen(occupation, gend, args.output_folder)
                else:
                    model_gen(occupation, gend, args.model, args.output_folder)
            except APIRequestFailedError as e:
               # 打印错误信息
                print(f"API请求失败: {e.message if hasattr(e, 'message') else e}")
                # 根据错误代码进行不同的处理
                # if e.error.code == "1301":
                #     print(f"在处理 {occupation} 和 {gend} 时发生错误: {e}")
                # 可以添加更多的错误代码处理逻辑
                # 然后决定是否继续循环或退出
                continue

if __name__ == "__main__":
    main()