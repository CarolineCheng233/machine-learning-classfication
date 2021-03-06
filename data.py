import numpy as np
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor
import torch
import nltk
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import gensim.downloader as api


def read():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    args = parser.parse_args()
    train_file = args.train_file
    data = pd.read_csv(train_file)
    # data.keys() = ['age', 'body type', 'bust size', 'category', 'height', 'item_id',
    #                'rating', 'rented for', 'review_date', 'review_summary', 'review_text',
    #                'size', 'user_id', 'weight', 'fit']
    # fit: large, fit, small


# 封装类
# class DataPrecessForSingleSentence:
#     def __init__(self, max_workers=10):
#         """
#         bert_tokenizer :分词器
#         dataset        :包含列名为'text'与'label'的pandas dataframe
#         """
#         self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
#         # 创建多线程池
#         # self.pool = ThreadPoolExecutor(max_workers=max_workers)
#
#     def get_input(self, sentence, max_seq_len=15):
#         """
#         通过多线程（因为notebook中多进程使用存在一些问题）的方式对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
#
#         入参:
#             sentences     : pandas的dataframe格式.
#             max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
#
#         出参:
#             seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
#             seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
#                           那么取值为1，否则为0。
#             seq_segment : shape等于seq，因为是单句，所以取值都为0。
#             labels      : 标签取值为{0,1,2}。
#         """
#         # 切词
#         tokens_seq = self.bert_tokenizer.tokenize(sentence)
#         # 获取定长序列及其mask
#         result = self.trunate_and_pad(tokens_seq, max_seq_len)
#         seq = [result[0]]
#         seq_mask = [result[1]]
#         seq_segment = [result[2]]
#         return torch.tensor(seq), torch.tensor(seq_mask), torch.tensor(seq_segment)
#
#     def trunate_and_pad(self, seq, max_seq_len):
#         """
#         1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
#            因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
#         2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
#
#         入参:
#             seq         : 输入序列，在本处其为单个句子。
#             max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
#
#         出参:
#             seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
#             seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
#                           那么取值为1，否则为0。
#             seq_segment : shape等于seq，因为是单句，所以取值都为0。
#         """
#         # 对超长序列进行截断
#         if len(seq) > (max_seq_len - 2):
#             seq = seq[0:(max_seq_len - 2)]
#         # 分别在首尾拼接特殊符号
#         seq = ['[CLS]'] + seq + ['[SEP]']
#         # ID化
#         seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
#         # 根据max_seq_len与seq的长度产生填充序列
#         padding = [0] * (max_seq_len - len(seq))
#         # 创建seq_mask
#         seq_mask = [1] * len(seq) + padding
#         # 创建seq_segment
#         seq_segment = [0] * len(seq) + padding
#         # 对seq拼接填充序列
#         seq += padding
#         assert len(seq) == max_seq_len
#         assert len(seq_mask) == max_seq_len
#         assert len(seq_segment) == max_seq_len
#         return seq, seq_mask, seq_segment


class DataProcessPipeline:

    def __init__(self, tokenizer_dir, allowed_keys=["body type"]):
        super().__init__()
        if 'fit' in allowed_keys:
            allowed_keys.remove('fit')
        self.allowed_keys = allowed_keys
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        # self.tokenizer = DataPrecessForSingleSentence()

    def __call__(self, data):
        result = {}
        if "review_text" in data:
            text = data["review_text"]
            if pd.isna(text):
                text = 'nan'
            result['text'] = self.tokenizer(text, truncation=True, max_length=64,
                                            padding="max_length", return_tensors="pt")
        return result


class Word2VecPipeline:

    def __init__(self):
        self.tokenizer = api.load('word2vec-google-news-300')

    def __call__(self, data):
        result = {}
        if "review_text" in data:
            text = data["review_text"]
            if pd.isna(text):
                text = 'nan'
            vector = np.zeros(300, dtype=np.float32)
            for word in text.split():
                try:
                    word_vec = self.tokenizer[word]
                except:
                    word_vec = np.zeros(300)
                vector += word_vec
            vector = (vector / len(text.split())).astype(np.float32)
            result['text'] = vector

        return result


class ItemDataset(Dataset):
    def __init__(self, file_name, pipeline, with_label=False, allowed_keys=["body type"]):
        if 'fit' in allowed_keys:
            allowed_keys.remove('fit')
        self.file_name = file_name
        self.pipeline = pipeline
        self.with_label = with_label
        self.allowed_keys = allowed_keys
        self.data = pd.read_csv(file_name)
        self.data_by_keys = dict()
        for key in allowed_keys:
            self.data_by_keys[key] = self.data[key].values
        self.label2idx = dict(large=0, fit=1, small=2)
        self.idx2label = {0: 'large', 1: 'fit', 2: 'small'}
        if with_label:
            self.labels = self.data['fit'].values
            self.one_hot_labels = []
            for label in self.labels:
                self.one_hot_labels.append(self.label2idx[label])
            self.one_hot_labels = np.array(self.one_hot_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = dict()
        for key in self.allowed_keys:
            result[key] = self.data_by_keys[key][idx]
        if self.with_label:
            return self.pipeline(result), self.one_hot_labels[idx]
        else:
            return self.pipeline(result)
