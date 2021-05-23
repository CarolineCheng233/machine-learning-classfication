import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


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
    val = "data/train_split_2.txt"
    file = pd.read_csv(val)
    data = file.values
    label = file['fit'].values
    print(len(data))
    print(sum(label == 'small') / len(data))
    print(sum(label == 'fit') / len(data))
    print(sum(label == 'large') / len(data))


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


def split_train_val_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    args = parser.parse_args()
    file, train_file, val_file, test_file = args.file, args.train_file, args.val_file, args.test_file
    ratio = np.array([0.7, 0.8, 1.0])
    file = pd.read_csv(file)
    data = file.values
    np.random.shuffle(data)
    keys = file.keys()
    numbers = (ratio * len(data)).astype(np.int)
    train = pd.DataFrame(data[:numbers[0]], columns=keys)
    val = pd.DataFrame(data[numbers[0]:numbers[1]], columns=keys)
    test = pd.DataFrame(data[numbers[1]:], columns=keys)
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)


if __name__ == '__main__':
    read_val()
