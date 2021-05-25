import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from transformers import pipeline
from sklearn.metrics import f1_score
from mmcv import ProgressBar
import gensim.downloader as api
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


def val_split():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--number', type=int, default=7544)
    parser.add_argument('--val', type=str, required=True)
    args = parser.parse_args()
    test, val, label, number = args.test, args.val, args.label, args.number
    test = pd.read_csv(test)
    data = test.values
    total = len(data)
    with open(label, 'r', encoding='utf-8') as f:
        label = np.array([line.strip() for line in f])
    choices = np.random.choice(total, number, replace=False)
    val_data = data[choices]
    val_label = label[choices].reshape(number, 1)
    data = np.concatenate((val_data, val_label), axis=1)
    keys = list(test.keys().values)
    keys.append('fit')
    keys = np.array(keys)
    df = pd.DataFrame(data, columns=pd.Index(keys))
    df.to_csv(val, index=False)


def read_val():
    val = "data/train_split.txt"
    file = pd.read_csv(val)
    # data = file.values
    import pdb; pdb.set_trace()
    # label = file['fit'].values
    # print(len(data))
    # print(sum(label == 'small') / len(data))
    # print(sum(label == 'fit') / len(data))
    # print(sum(label == 'large') / len(data))


def train_split():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--number', type=int, default=10000)
    parser.add_argument('--train_split', type=str, required=True)
    args = parser.parse_args()
    train, train_split, number = args.train, args.train_split, args.number
    train = pd.read_csv(train)
    data = train.values
    total = len(data)
    choices = np.random.choice(total, number, replace=False)
    data = data[choices]
    df = pd.DataFrame(data, columns=train.keys())
    df.to_csv(train_split, index=False)


def integrate_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    test, label, output = args.test, args.label, args.output
    test = pd.read_csv(test)
    data = test.values
    number = len(data)
    with open(label, 'r', encoding='utf-8') as f:
        label = np.array([line.strip() for line in f]).reshape(number, 1)
    data = np.concatenate((data, label), axis=1)
    keys = list(test.keys().values)
    keys.append('fit')
    keys = np.array(keys)
    df = pd.DataFrame(data, columns=pd.Index(keys))
    df.to_csv(output, index=False)


def split_train_label():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    train, output = args.train, args.output
    train = pd.read_csv(train)
    label = train['fit'].values
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label))


def resample():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--number', type=int, required=True)
    args = parser.parse_args()
    train, output, number = args.train, args.output, args.number
    number = number // 3
    train = pd.read_csv(train)
    data = train.values
    label = train['fit'].values
    split = defaultdict(list)
    for i, l in enumerate(label):
        split[l].append(i)
    split['small'] = np.random.choice(split['small'], number, replace=False)
    split['fit'] = np.random.choice(split['fit'], number, replace=False)
    split['large'] = np.random.choice(split['large'], number, replace=False)
    small = data[split['small']]
    fit = data[split['fit']]
    large = data[split['large']]
    data = np.concatenate((small, fit, large), axis=0)
    np.random.shuffle(data)

    keys = list(train.keys().values)
    keys = np.array(keys)
    df = pd.DataFrame(data, columns=pd.Index(keys))
    df.to_csv(output, index=False)


def split_train_val():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    args = parser.parse_args()
    file, train_file, val_file = args.file, args.train_file, args.val_file
    ratio = np.array([0.8, 1.0])
    file = pd.read_csv(file)
    data = file.values
    np.random.shuffle(data)
    keys = file.keys()
    numbers = (ratio * len(data)).astype(np.int)
    train = pd.DataFrame(data[:numbers[0]], columns=keys)
    val = pd.DataFrame(data[numbers[0]:numbers[1]], columns=keys)
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)


def get_text_length():
    data = pd.read_csv("data/train_split.txt")
    summary = data['review_summary'].values
    import pdb; pdb.set_trace()
    lengths = [len(text.split()) if isinstance(text, str) else len('nan'.split()) for text in summary]
    avg = sum(lengths) / len(lengths)
    maximum = max(lengths)
    print(avg)
    print(maximum)


def classify_by_text_match(text, label=None):
    small = re.compile(r'small')
    large = re.compile(r'large')
    sent_classifier = pipeline('sentiment-analysis')
    str2idx = dict(small=0, fit=1, large=2)
    idx2str = {0: 'small', 1: 'fit', 2: 'large'}
    result_list = []
    pb = ProgressBar(len(text))
    pb.start()
    for t in text:
        if not pd.isna(t):
            result = sent_classifier(t)[0]
            sent, score = result['label'], result['score']
            if sent == 'POSITIVE' and score > 0.3:
                result_list.append(str2idx['fit'])
            else:
                if len(small.findall(t)) > 0:
                    result_list.append(str2idx['small'])
                elif len(large.findall(t)) > 0:
                    result_list.append(str2idx['large'])
                else:
                    result_list.append(np.random.randint(0, 3))
        else:
            result_list.append(np.random.randint(0, 3))
        pb.update()
    result_list = np.array(result_list)
    print()
    if label is not None:
        idx_label = []
        for l in label:
            idx_label.append(str2idx[l])
        idx_label = np.array(idx_label)
        f1 = f1_score(idx_label, result_list, average='macro')
        print(f'f1 score = {f1}')
    else:
        wfile = "data/output.txt"
        str_result = []
        for r in result_list:
            str_result.append(idx2str[r])
        with open(wfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(str_result))


def train_word_vec(text):
    # data = []
    # for i in sent_tokenize(text):
    #     temp = []
    #     import pdb; pdb.set_trace()
    #     # tokenize the sentence into words
    #     for j in word_tokenize(i):
    #         temp.append(j.lower())
    #
    #     data.append(temp)
    # model1 = gensim.models.Word2Vec(data, min_count=1,
    #                                 size=100, window=5)
    pass


def get_word2vec():
    model = api.load("glove-twitter-25")
    vector = model.wv.get_vector('test')
    print(vector)


def foo():
    print('foo')
    model = api.load('word2vec-google-news-300')
    # Word2Vec(sentences, min_count=1)
    # vector = model.wv('test')
    vector = model('test')
    print(vector.shape)


def read_data(file, label=False):
    data = pd.read_csv(file)
    if label:
        return data['review_text'].values, data['fit'].values
    else:
        return data['review_text'].values


if __name__ == '__main__':
    # data, label = read_data('data/val_split.txt', True)
    # classify_by_text_match(data, label)
    # nltk.download('punkt')
    # train = read_data('data/train.txt')
    # test = read_data('data/test.txt')
    # total = np.concatenate((train, test))
    # train_word_vec(total)
    # get_word2vec()
    # print(total.shape)
    # foo()
    get_word2vec()
