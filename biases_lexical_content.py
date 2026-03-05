import spacy
from spacy.matcher import Matcher
from collections import Counter
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from spacy.lang.zh.stop_words import STOP_WORDS as zh_stop_words

def is_chinese(word):
    """判断字符串是否全部由中文字符组成"""
    return all('\u4e00' <= char <= '\u9fff' for char in word)

class Word_Extraction:
    def __init__(self, word_types=None):
        self.nlp = spacy.load("zh_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        patterns = []
        
        self.word_type_map = {
            'noun': 'NOUN',
            'adj': 'ADJ',
            'verb': 'VERB',
            'adv': 'ADV'
        }
        
        if word_types:
            for word_type in word_types:
                if word_type in self.word_type_map:
                    patterns.append([{'POS': self.word_type_map[word_type]}])
        self.matcher.add("word_extraction", patterns)

    def extract_word(self, doc):
        doc = self.nlp(doc)
        matches = self.matcher(doc)
        vocab = []
        for match_id, start, end in matches:
            span = doc[start:end]
            # 使用token的pos属性而不是span的pos_
            token = doc[start]
            if is_chinese(span.text):  # 只保留纯中文词汇
                vocab.append((span.text, token.pos_))
        return vocab

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/deepseek/cbg/all_2_para_w_deepseek.csv")
    # parser.add_argument('-f', '--file_name', type=str, default="./generated_letters/deepseek/cbg/clg_letters.csv")
    parser.add_argument('-m', '--model_type', default='deepseek', required=False)
    parser.add_argument('-t', '--threshold', default=250, required=False)
    parser.add_argument('-o', '--output_dir', type=str, default="./result/lexical_analysis", help="输出目录")
    args = parser.parse_args()
    
    rec_letters = pd.read_csv(args.file_name)
    INPUT = "{}_gen".format(args.model_type)
    all_texts = rec_letters[INPUT].tolist()

    # 初始化词性提取器
    noun_extract = Word_Extraction(['noun'])
    adj_extract = Word_Extraction(['adj'])
    
    # 存储分类词汇
    word_dict = {
        'noun': Counter(),
        'adj': Counter()
    }
    
    # 处理所有文本
    for text in tqdm(all_texts, desc="处理文本"):
        if pd.isna(text) or not isinstance(text, str):  # 跳过空值和非字符串
            continue
        for word, pos in noun_extract.extract_word(text):
            word = word.strip('，。、！？；：“”‘’（）《》【】').lower()
            if word not in zh_stop_words:
                word_dict['noun'][word] += 1
                
        for word, pos in adj_extract.extract_word(text):
            word = word.strip('，。、！？；：“”‘’（）《》【】').lower()
            if word not in zh_stop_words:
                word_dict['adj'][word] += 1
                
    # 输出高频词汇
    threshold = int(args.threshold)
    
    output_data = []
    all_output_data = []

    print("\n高频名词：")
    for word, count in word_dict['noun'].most_common():
        all_output_data.append({'词性': '名词', '词汇': word, '频次': count})
        if count >= threshold:
            print(f"{word}: {count}")
            output_data.append({'词性': '名词', '词汇': word, '频次': count})
            
    print("\n高频形容词：")
    for word, count in word_dict['adj'].most_common():
        all_output_data.append({'词性': '形容词', '词汇': word, '频次': count})
        if count >= threshold:
            print(f"{word}: {count}")
            output_data.append({'词性': '形容词', '词汇': word, '频次': count})

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    input_basename = os.path.basename(args.file_name).split('.')[0]

    # 保存高频结果
    if output_data:
        output_file = os.path.join(args.output_dir, f"{input_basename}_lexical_stats.csv")
        df_out = pd.DataFrame(output_data)
        df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n高频结果已保存至: {output_file}")
        
    # 保存全量结果
    if all_output_data:
        output_file_all = os.path.join(args.output_dir, f"{input_basename}_lexical_stats_all.csv")
        df_out_all = pd.DataFrame(all_output_data)
        df_out_all.to_csv(output_file_all, index=False, encoding='utf-8-sig')
        print(f"全量结果已保存至: {output_file_all}")

   