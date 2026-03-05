import spacy
import pandas as pd
from tqdm import tqdm
from spacy.matcher import Matcher
from collections import Counter
from operator import itemgetter
import scipy.stats as stats
from argparse import ArgumentParser
import word_constants
import jieba
import os  # 用于处理文件路径

if __name__ == '__main__':
    """
    Arguments:
    file_name: Directory of the input file.
    - For analyzing CLG letters, pass in './generated_letters/{model_type}/clg/clg_letters.csv'
    model_type: Model used to generated the letters.
    """
    parser = ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/zhipuai/clg/clg_letters.csv")
    # parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/deepseek/clg/clg_letters.csv")
    # parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/deepseek/cbg/all_2_para_w_deepseek.csv")
    # parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/zhipuai/cbg/all_2_para_w_zhipuai.csv")
    parser.add_argument('-m', '--model_type', default='zhipuai', required=False)
    parser.add_argument('-o', '--output_dir', type=str, default="./result/lexical_analysis", help="输出目录")
    # parser.add_argument('-m', '--model_type', default='deepseek', required=False)
    args = parser.parse_args()
    rec_letters = pd.read_csv(args.file_name)
    INPUT = "{}_gen".format(args.model_type)

    if rec_letters['gender'][0] in ['男性', '女性']:
        rec_letters_m = rec_letters[rec_letters['gender'] == '男性']
        rec_letters_f = rec_letters[rec_letters['gender'] == '女性']
    elif rec_letters['gender'][0] in ['male', 'female']:
        rec_letters_m = rec_letters[rec_letters['gender'] == 'male']
        rec_letters_f = rec_letters[rec_letters['gender'] == 'female']
    elif rec_letters['gender'][0] in ['m', 'f']:
        rec_letters_m = rec_letters[rec_letters['gender'] == 'm']
        rec_letters_f = rec_letters[rec_letters['gender'] == 'f']
    else:
        rec_letters_m = rec_letters[rec_letters['gender'] == '男']
        rec_letters_f = rec_letters[rec_letters['gender'] == '女']
    
    # if rec_letters['gender'][0] in ['male', 'female']:
    #     rec_letters_m = rec_letters[rec_letters['gender'] == 'male']
    #     rec_letters_f = rec_letters[rec_letters['gender'] == 'female']
    # else:
    #     rec_letters_m = rec_letters[rec_letters['gender'] == 'm']
    #     rec_letters_f = rec_letters[rec_letters['gender'] == 'f']

    # generated letters
    rec_letters_m = rec_letters_m[INPUT].tolist()
    rec_letters_f = rec_letters_f[INPUT].tolist()

    effort_f, excellence_f, agentic_f, communal_f, career_f, family_f = 0, 0, 0, 0, 0, 0
    effort_m, excellence_m, agentic_m, communal_m, career_m, family_m = 0, 0, 0, 0, 0, 0

    all_f, all_m = 0, 0

    for i in tqdm(range(len(rec_letters_f)), ascii=True):
        text = str(rec_letters_f[i]) if pd.notna(rec_letters_f[i]) else ""
        # 打印当前处理的文本
        # print(f"Processing female text: {text}")
        n = len(text)
        all_f += n
        for w in word_constants.effort_words:
            if w in text:
                effort_f += 1
                # print(f"Matched effort word: {w}")
        for w in word_constants.excellence_words:
            if w in text:
                excellence_f += 1
        for w in word_constants.agentic_words:
            if w in text:
                agentic_f += 1
        for w in word_constants.communal_words:
            if w in text:
                communal_f += 1
        for w in word_constants.career_words:
            if w in text:
                career_f += 1
        for w in word_constants.personal_words:
            if w in text:
                family_f += 1

    for i in tqdm(range(len(rec_letters_m)), ascii=True):
        text = str(rec_letters_m[i]) if pd.notna(rec_letters_m[i]) else ""
        # 打印当前处理的文本
        # print(f"Processing male text: {text}")
        n = len(text)
        all_m += n
        for w in word_constants.effort_words:
            if w in text:
                effort_m += 1
                # print(f"Matched effort word: {w}")
        for w in word_constants.excellence_words:
            if w in text:
                excellence_m += 1
        for w in word_constants.agentic_words:
            if w in text:
                agentic_m += 1
        for w in word_constants.communal_words:
            if w in text:
                communal_m += 1
        for w in word_constants.career_words:
            if w in text:
                career_m += 1
        for w in word_constants.personal_words:
            if w in text:
                family_m += 1

    # For normal analysis
    small_number = 0.001
    effort_score = ((effort_m + small_number) / (all_m - effort_m + small_number)) / ((effort_f + small_number) / (all_f - effort_f + small_number))
    excellence_score = ((excellence_m + small_number) / (all_m - excellence_m + small_number)) / ((excellence_f + small_number) / (all_f - excellence_f + small_number))
    # masculine_score = ((masculine_m + small_number) / (all_m - masculine_m + small_number)) / ((masculine_f + small_number) / (all_f - masculine_f + small_number))
    # feminine_score = ((feminine_m + small_number) / (all_m - feminine_m + small_number)) / ((feminine_f + small_number) / (all_f - feminine_f + small_number))
    agentic_score = ((agentic_m + small_number) / (all_m - agentic_m + small_number)) / ((agentic_f + small_number) / (all_f - agentic_f + small_number))
    communal_score = ((communal_m + small_number) / (all_m - communal_m + small_number)) / ((communal_f + small_number) / (all_f - communal_f + small_number))
    career_score = ((career_m + small_number) / (all_m - career_m + small_number)) / ((career_f + small_number) / (all_f - career_f + small_number))
    family_score = ((family_m + small_number) / (all_m - family_m + small_number)) / ((family_f + small_number) / (all_f - family_f + small_number))
    # leader_score = ((leader_m + small_number) / (all_m - leader_m + small_number)) / ((leader_f + small_number) / (all_f - leader_f + small_number))

    # 打印结果
    print('\n effort: Male {}, Female {}, score {}'.format(effort_m, effort_f, effort_score))
    print('\n excellence: Male {}, Female {}, score {}'.format(excellence_m, excellence_f, excellence_score))
    # print('\n masculine: Male {}, Female {}, score {}'.format(masculine_m, masculine_f, masculine_score))
    # print('\n feminine: Male {}, Female {}, score {}'.format(feminine_m, feminine_f, feminine_score))
    print('\n agentic: Male {}, Female {}, score {}'.format(agentic_m, agentic_f, agentic_score))
    print('\n communal: Male {}, Female {}, score {}'.format(communal_m, communal_f, communal_score))
    print('\n career: Male {}, Female {}, score {}'.format(career_m, career_f, career_score))
    print('\n family: Male {}, Female {}, score {}'.format(family_m, family_f, family_score))
    # print('\n leadership: Male {}, Female {}, score {}'.format(leader_m, leader_f, leader_score))

    # 保存结果到文件
    output = {
        '词汇类别': ['卓越型特质', '努力型特质', '主体性特质', '共有性特质', '职业特质', '个人非职场特质'],
        '男性频次': [excellence_m, effort_m, agentic_m, communal_m, career_m, family_m],
        '女性频次': [excellence_f, effort_f, agentic_f, communal_f, career_f, family_f],
        '偏见分数': [excellence_score, effort_score, agentic_score, communal_score, career_score, family_score]
    }

    # 将字典转换为 DataFrame
    df = pd.DataFrame(output)

    # 指定保存路径
    os.makedirs(args.output_dir, exist_ok=True)  # 确保输出文件夹存在
    input_basename = os.path.basename(args.file_name).split('.')[0]
    output_file = os.path.join(args.output_dir, f'{input_basename}_bias_stats.csv')

    # 保存到 CSV 文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到文件: {output_file}")
